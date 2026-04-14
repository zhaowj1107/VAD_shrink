import json
import sys
import types
import wave
from array import array
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_silence(path: Path, duration_sec: float, sample_rate: int) -> None:
    samples = array("h", [0] * int(duration_sec * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _write_pcm24(path: Path, sample_rate: int, samples: list[int]) -> None:
    frames = bytearray()
    for sample in samples:
        frames.extend(int(sample).to_bytes(3, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(3)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def test_read_wav_mono_supports_pcm24(tmp_path):
    from vad_baseline.onnx_runtime import read_wav_mono

    audio_path = tmp_path / "pcm24.wav"
    _write_pcm24(audio_path, sample_rate=16000, samples=[0, 8388607])

    sample_rate, samples = read_wav_mono(audio_path)

    assert sample_rate == 16000
    assert np.allclose(samples, np.array([0.0, 8388607 / 8388608], dtype=np.float32))


def test_read_wav_mono_uses_torchcodec_when_available(monkeypatch):
    from vad_baseline.onnx_runtime import read_wav_mono

    class FakeAudioSamples:
        data = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32)
        sample_rate = 8000

    class FakeAudioDecoder:
        def __init__(self, source):
            assert source == "clip.wav"

        @staticmethod
        def get_all_samples():
            return FakeAudioSamples()

    fake_module = types.ModuleType("torchcodec.decoders")
    fake_module.AudioDecoder = FakeAudioDecoder
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_module)

    sample_rate, samples = read_wav_mono("clip.wav")

    assert sample_rate == 8000
    assert np.allclose(samples, np.array([0.25, -0.5, 0.75], dtype=np.float32))


def test_load_onnx_vad_runtime_reads_metadata_and_builds_session(tmp_path):
    from vad_baseline.onnx_runtime import load_onnx_vad_runtime

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    metadata_path = tmp_path / "model.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "sample_rate": 16000,
                "time_resolution": 0.01,
                "source_model_name": "speechbrain/vad-crdnn-libriparty",
                "input_names": ["feats"],
                "output_names": ["speech_probabilities"],
                "opset_version": 17,
                "frontend": "speechbrain_fbank",
            }
        )
    )
    calls = {}

    def fake_session_factory(onnx_model_path, providers):
        calls["onnx_model_path"] = onnx_model_path
        calls["providers"] = providers
        return "fake-session"

    def fake_feature_extractor_factory(metadata, model_path=None):
        calls["feature_metadata"] = metadata["frontend"]
        return "fake-feature-extractor"

    runtime = load_onnx_vad_runtime(
        model_path,
        session_factory=fake_session_factory,
        feature_extractor_factory=fake_feature_extractor_factory,
    )

    assert calls == {
        "feature_metadata": "speechbrain_fbank",
        "onnx_model_path": str(model_path),
        "providers": ["CPUExecutionProvider"],
    }
    assert runtime.session == "fake-session"
    assert runtime.feature_extractor == "fake-feature-extractor"
    assert runtime.model_path == str(model_path.resolve())
    assert runtime.sample_rate == 16000
    assert runtime.time_resolution == 0.01
    assert runtime.input_names == ["feats"]
    assert runtime.output_names == ["speech_probabilities"]


def test_load_onnx_vad_runtime_requires_metadata_sidecar(tmp_path):
    from vad_baseline.onnx_runtime import load_onnx_vad_runtime

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    with pytest.raises(FileNotFoundError, match="metadata"):
        load_onnx_vad_runtime(model_path, session_factory=lambda *args: None)


def test_onnx_vad_runtime_predict_segments_from_session_outputs(tmp_path):
    from vad_baseline.onnx_runtime import ONNXVADRuntime

    audio_path = tmp_path / "clip.wav"
    _write_silence(audio_path, duration_sec=1.0, sample_rate=100)

    class FakeSession:
        def run(self, output_names, inputs):
            assert output_names == ["speech_probabilities"]
            assert set(inputs) == {"feats"}
            assert inputs["feats"].shape == (2, 5, 40)
            return [
                np.array(
                    [
                        [[0.1], [0.6], [0.7], [0.2], [0.1], [0.0]],
                        [[0.1], [0.8], [0.9], [0.1], [0.1], [0.0]],
                    ],
                    dtype=np.float32,
                )
            ]

    def fake_feature_extractor(wavs):
        assert wavs.shape == (2, 50)
        return np.zeros((2, 5, 40), dtype=np.float32)

    runtime = ONNXVADRuntime(
        session=FakeSession(),
        model_path=str(tmp_path / "model.onnx"),
        sample_rate=100,
        time_resolution=0.1,
        input_names=["feats"],
        output_names=["speech_probabilities"],
        feature_extractor=fake_feature_extractor,
        large_chunk_size=1.0,
        small_chunk_size=0.5,
        close_th=0.0,
        len_th=0.0,
        speech_th=0.5,
        double_check=False,
    )

    assert runtime.predict_segments(audio_path) == [
        {"start": 0.1, "end": 0.2, "duration": 0.1},
        {"start": 0.6, "end": 0.7, "duration": 0.1},
    ]


def test_numpy_fbank_shape_and_dtype():
    from vad_baseline.onnx_runtime import _numpy_fbank

    mel_matrix = np.random.rand(201, 40).astype(np.float32)
    window = np.hamming(400).astype(np.float32)
    wavs = np.zeros((2, 16000), dtype=np.float32)

    feats = _numpy_fbank(wavs, mel_matrix, window, hop_length=160)

    assert feats.shape == (2, 101, 40)  # (batch, frames, mels)
    assert feats.dtype == np.float32


def test_numpy_fbank_1d_input_accepted():
    from vad_baseline.onnx_runtime import _numpy_fbank

    mel_matrix = np.random.rand(201, 40).astype(np.float32)
    window = np.hamming(400).astype(np.float32)
    wav = np.zeros(16000, dtype=np.float32)

    feats = _numpy_fbank(wav, mel_matrix, window, hop_length=160)

    assert feats.shape == (1, 101, 40)


def test_build_feature_extractor_numpy_fbank(tmp_path):
    from vad_baseline.onnx_runtime import build_feature_extractor_from_metadata

    mel_matrix = np.ones((5, 3), dtype=np.float32)
    window = np.ones(8, dtype=np.float32)
    np.savez(str(tmp_path / "model.fbank.npz"), mel_matrix=mel_matrix, window=window)

    metadata = {"frontend": "numpy_fbank", "hop_length": 4}
    extractor = build_feature_extractor_from_metadata(
        metadata, model_path=str(tmp_path / "model.onnx")
    )

    wavs = np.zeros((1, 32), dtype=np.float32)
    feats = extractor(wavs)
    assert feats.shape[0] == 1
    assert feats.shape[2] == 3  # n_mels
    assert feats.dtype == np.float32


def test_build_feature_extractor_numpy_fbank_raises_without_model_path():
    from vad_baseline.onnx_runtime import build_feature_extractor_from_metadata

    with pytest.raises(ValueError, match="model_path"):
        build_feature_extractor_from_metadata({"frontend": "numpy_fbank"})
