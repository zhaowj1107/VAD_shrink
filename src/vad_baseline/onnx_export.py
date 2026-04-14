import json
from pathlib import Path

import numpy as np
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


def fbank_path_for_model(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.fbank.npz")


def extract_fbank_weights(vad_model):
    """Extract mel filterbank matrix and window from the loaded SpeechBrain VAD model.

    Returns (mel_matrix, window) as float32 numpy arrays:
      mel_matrix: (n_fft//2+1, n_mels) — linear mel filterbank weights
      window:     (win_length,)         — analysis window (hamming)
    """
    fb = vad_model.mods.compute_features

    # Hamming window — stored as a buffer in the STFT submodule.
    stft_mod = getattr(fb, "compute_STFT", None)
    window = None
    if stft_mod is not None:
        for attr in ("window", "win"):
            w = getattr(stft_mod, attr, None)
            if w is not None and hasattr(w, "detach"):
                window = w.detach().cpu().numpy().astype(np.float32)
                break

    if window is None:
        # Fallback: reconstruct from known parameters (n_fft=400, hamming).
        n = 400
        window = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(
            np.float32
        )

    # Mel filterbank matrix — SpeechBrain's Filterbank recomputes this each
    # forward pass from f_central and band (it is NOT stored as a buffer).
    # We reconstruct it once using the same logic as Filterbank.forward but
    # skipping the log/dB step, giving us the linear (201, 40) weight matrix.
    fbank_mod = getattr(fb, "compute_fbanks", None)
    if fbank_mod is None:
        raise RuntimeError(
            "Cannot find compute_fbanks on vad_model.mods.compute_features; "
            "inspect the model and update extract_fbank_weights accordingly."
        )
    with torch.no_grad():
        n_stft = int(fbank_mod.n_stft)  # 201
        f_central_mat = fbank_mod.f_central.repeat(n_stft, 1).transpose(0, 1)
        band_mat = fbank_mod.band.repeat(n_stft, 1).transpose(0, 1)
        mel_matrix = (
            fbank_mod._create_fbank_matrix(f_central_mat, band_mat)
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # shape: (n_stft, n_mels) = (201, 40)

    return mel_matrix, window


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

    # Extract and save the numpy fbank sidecar so inference requires no PyTorch.
    mel_matrix, window = extract_fbank_weights(vad_model)
    fbank_path = fbank_path_for_model(output_path)
    np.savez(str(fbank_path), mel_matrix=mel_matrix, window=window)

    metadata = {
        "source_model_name": model_source_name(),
        "sample_rate": int(vad_model.sample_rate),
        "time_resolution": float(vad_model.time_resolution),
        "input_names": ["feats"],
        "output_names": list(OUTPUT_NAMES),
        "opset_version": int(opset_version),
        "frontend": "numpy_fbank",
        "hop_length": 160,
    }
    metadata_path = metadata_path_for_model(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "model_path": str(output_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "fbank_path": str(fbank_path.resolve()),
        **metadata,
    }
