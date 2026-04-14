import csv
import sys
from types import MethodType

import torch
import torch.ao.quantization as tq
from torch.ao.quantization import quantize_dynamic

from vad_baseline.backends.speechbrain_fp32 import SpeechBrainFP32Backend
from vad_baseline.model import load_vad_model


def _ensure_quantized_rnn_compat(modules):
    if not hasattr(modules, "modules"):
        return modules
    for module in modules.modules():
        module_type = type(module)
        if (
            module_type.__module__
            == "torch.ao.nn.quantized.dynamic.modules.rnn"
            and module_type.__name__ == "GRU"
            and not hasattr(module, "flatten_parameters")
        ):
            module.flatten_parameters = MethodType(lambda self: None, module)
    return modules


def _audio_paths_from_manifest(manifest_path):
    paths = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["audio_path"])
    return paths


# Layers that static quantization on CPU (qnnpack) does not support.
_SKIP_STATIC_QUANT = (torch.nn.LayerNorm,)


class _StaticQuantWrapper(torch.nn.Module):
    """Adds QuantStub/DeQuantStub around a module so tq.prepare/convert work
    correctly: input float → quant → inner → dequant → output float."""

    def __init__(self, inner):
        super().__init__()
        self.quant = tq.QuantStub()
        self.inner = inner
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        return self.dequant(self.inner(self.quant(x)))


def _prepare_static_wrapper(wrapper, qconfig):
    wrapper.qconfig = qconfig
    for module in wrapper.modules():
        if isinstance(module, _SKIP_STATIC_QUANT):
            module.qconfig = None
    tq.prepare(wrapper, inplace=True)


def _fuse_dnn_linear_bn(dnn):
    """Absorb BatchNorm1d into the preceding Linear for each DNN_Block.

    SpeechBrain wraps layers as block.linear.w (Linear) and block.norm.norm
    (BatchNorm1d).  After fusion, block.linear.w becomes a plain Linear with
    BN parameters baked in, and block.norm.norm becomes Identity.  This lets
    static INT8 quantize Linear without quantized → float BN mismatches.
    """
    from torch.nn.utils.fusion import fuse_linear_bn_eval

    for _name, block in dnn.named_children():
        try:
            linear_leaf = block.linear.w
            bn_leaf = block.norm.norm
        except AttributeError:
            continue
        if isinstance(linear_leaf, torch.nn.Linear) and isinstance(
            bn_leaf, torch.nn.BatchNorm1d
        ):
            block.linear.w = fuse_linear_bn_eval(linear_leaf, bn_leaf)
            block.norm.norm = torch.nn.Identity()


class SpeechBrainStaticINT8Backend(SpeechBrainFP32Backend):
    """Static INT8 quantization: CNN and DNN quantized statically via calibration,
    GRU quantized dynamically (static GRU is not well-supported in PyTorch)."""

    backend_name = "speechbrain_static_int8"
    model_name = "speechbrain/vad-crdnn-libriparty-static-int8"

    def __init__(self, calibration_manifest_path, run_opts=None):
        super().__init__(run_opts)
        self.calibration_manifest_path = str(calibration_manifest_path)

    def load(self):
        if sys.platform == "darwin":
            torch.backends.quantized.engine = "qnnpack"

        vad_model = load_vad_model(self.run_opts)
        vad_model.eval()

        # Absorb BatchNorm1d into Linear in DNN blocks before quantization.
        _fuse_dnn_linear_bn(vad_model.mods.dnn)

        # --- Static INT8: DNN only ---
        # CNN uses Conv+LayerNorm throughout.  torch.nn.LayerNorm internally
        # calls native_batch_norm and rejects quantized-tensor input in
        # PyTorch eager mode, so CNN cannot be naively statically quantized
        # without explicit dequant/requant stubs around every LayerNorm.
        # We leave CNN as FP32 and only apply static quantization to the DNN.
        dnn_wrapper = _StaticQuantWrapper(vad_model.mods.dnn)
        vad_model.mods.dnn = dnn_wrapper

        qconfig = tq.get_default_qconfig("qnnpack")
        _prepare_static_wrapper(dnn_wrapper, qconfig)

        # Calibration: run full pipeline on provided audio files
        audio_paths = _audio_paths_from_manifest(self.calibration_manifest_path)
        with torch.no_grad():
            for audio_path in audio_paths:
                vad_model.get_speech_prob_file(audio_path)

        tq.convert(dnn_wrapper, inplace=True)

        # --- Dynamic INT8: GRU ---
        vad_model.mods.rnn = _ensure_quantized_rnn_compat(
            quantize_dynamic(vad_model.mods.rnn, {torch.nn.GRU}, dtype=torch.qint8)
        )

        return vad_model
