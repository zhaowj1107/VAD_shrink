import json
from pathlib import Path

import torch

from vad_baseline.model import load_vad_model, model_source_name

DEFAULT_ONNX_OPSET = 17
DEFAULT_SAMPLE_FRAMES = 160000
INPUT_NAMES = ["wavs", "wav_lens"]
OUTPUT_NAMES = ["speech_probabilities"]


def normalize_sentence_norm_for_export(feats, mean_var_norm):
    if getattr(mean_var_norm, "mean_norm", True):
        current_mean = torch.mean(feats, dim=1, keepdim=True)
    else:
        current_mean = torch.zeros_like(feats[:, :1, ...])

    if getattr(mean_var_norm, "std_norm", True):
        current_std = torch.std(feats, dim=1, keepdim=True)
    else:
        current_std = torch.ones_like(feats[:, :1, ...])

    eps = float(getattr(mean_var_norm, "eps", 1e-10))
    current_std = torch.maximum(
        current_std,
        eps * torch.ones_like(current_std),
    )
    return (feats - current_mean.detach()) / current_std.detach()


def can_use_static_sentence_norm(mean_var_norm):
    return (
        getattr(mean_var_norm, "norm_type", None) == "sentence"
        and not getattr(mean_var_norm, "training", False)
    )


class SpeechBrainChunkExportWrapper(torch.nn.Module):
    def __init__(self, vad_model):
        super().__init__()
        self.compute_features = vad_model.mods.compute_features
        self.mean_var_norm = vad_model.mods.mean_var_norm
        self.cnn = vad_model.mods.cnn
        self.rnn = vad_model.mods.rnn
        self.dnn = vad_model.mods.dnn

    def forward(self, wavs, wav_lens):
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        wavs = wavs.float()
        wav_lens = torch.ones(
            (wavs.shape[0],),
            dtype=torch.float32,
            device=wavs.device,
        )

        feats = self.compute_features(wavs)
        if can_use_static_sentence_norm(self.mean_var_norm):
            feats = normalize_sentence_norm_for_export(
                feats,
                self.mean_var_norm,
            )
        else:
            feats = self.mean_var_norm(feats, wav_lens)
        outputs = self.cnn(feats)
        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )
        outputs, _ = self.rnn(outputs)
        outputs = self.dnn(outputs)
        return torch.sigmoid(outputs)


class SpeechBrainCoreExportWrapper(torch.nn.Module):
    def __init__(self, vad_model):
        super().__init__()
        self.mean_var_norm = vad_model.mods.mean_var_norm
        self.cnn = vad_model.mods.cnn
        self.rnn = vad_model.mods.rnn
        self.dnn = vad_model.mods.dnn

    def forward(self, feats):
        feats = feats.float()
        if can_use_static_sentence_norm(self.mean_var_norm):
            feats = normalize_sentence_norm_for_export(
                feats,
                self.mean_var_norm,
            )
        else:
            wav_lens = torch.ones(
                (feats.shape[0],),
                dtype=torch.float32,
                device=feats.device,
            )
            feats = self.mean_var_norm(feats, wav_lens)

        outputs = self.cnn(feats)
        outputs = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )
        outputs, _ = self.rnn(outputs)
        outputs = self.dnn(outputs)
        return torch.sigmoid(outputs)


def metadata_path_for_model(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.metadata.json")


def export_speechbrain_onnx(
    output_path,
    run_opts=None,
    load_model_fn=load_vad_model,
    export_fn=None,
    sample_frames=DEFAULT_SAMPLE_FRAMES,
    opset_version=DEFAULT_ONNX_OPSET,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_fn = export_fn or torch.onnx.export

    vad_model = load_model_fn(run_opts)
    wrapper = SpeechBrainCoreExportWrapper(vad_model)
    wrapper.eval()

    dummy_wavs = torch.zeros((1, sample_frames), dtype=torch.float32)
    with torch.no_grad():
        dummy_feats = vad_model.mods.compute_features(dummy_wavs)

    export_fn(
        wrapper,
        (dummy_feats,),
        output_path,
        input_names=["feats"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "feats": {0: "batch_size", 1: "num_frames"},
            "speech_probabilities": {0: "batch_size", 1: "num_frames"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    metadata = {
        "source_model_name": model_source_name(),
        "sample_rate": int(vad_model.sample_rate),
        "time_resolution": float(vad_model.time_resolution),
        "input_names": ["feats"],
        "output_names": list(OUTPUT_NAMES),
        "opset_version": int(opset_version),
        "frontend": "speechbrain_fbank",
    }
    metadata_path = metadata_path_for_model(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "model_path": str(output_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        **metadata,
    }
