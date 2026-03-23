from types import MethodType

import torch
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


class SpeechBrainDynamicINT8Backend(SpeechBrainFP32Backend):
    backend_name = "speechbrain_dynamic_int8"
    model_name = "speechbrain/vad-crdnn-libriparty-dynamic-int8"

    def load(self):
        vad_model = load_vad_model(self.run_opts)
        vad_model.mods = _ensure_quantized_rnn_compat(
            quantize_dynamic(
                vad_model.mods,
                {torch.nn.GRU, torch.nn.Linear},
                dtype=torch.qint8,
            )
        )
        return vad_model
