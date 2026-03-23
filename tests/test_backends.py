import math
import sys
import types
import wave
from array import array
from pathlib import Path

import torch

from vad_baseline.backends import get_backend, list_backend_names
from vad_baseline.backends.speechbrain_fp32 import SpeechBrainFP32Backend


def _write_waveform(
    path: Path,
    chunks,
    sample_rate: int = 16000,
):
    samples = array("h")
    for duration_sec, amplitude, frequency_hz in chunks:
        num_samples = int(duration_sec * sample_rate)
        if not frequency_hz:
            samples.extend([0] * num_samples)
            continue

        for sample_index in range(num_samples):
            sample_value = int(
                32767
                * amplitude
                * math.sin(
                    2
                    * math.pi
                    * frequency_hz
                    * (sample_index / sample_rate)
                )
            )
            samples.append(sample_value)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_list_backend_names_includes_speechbrain_fp32():
    assert "speechbrain_fp32" in list_backend_names()


def test_get_backend_returns_speechbrain_backend():
    backend = get_backend("speechbrain_fp32")

    assert isinstance(backend, SpeechBrainFP32Backend)
    assert backend.backend_name == "speechbrain_fp32"
    assert backend.model_name == "speechbrain/vad-crdnn-libriparty"


def test_speechbrain_backend_load_uses_model_loader(monkeypatch):
    calls = {}
    backend = SpeechBrainFP32Backend(run_opts={"device": "cpu"})

    def fake_load_vad_model(run_opts=None):
        calls["run_opts"] = run_opts
        return "fake-model"

    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_fp32.load_vad_model",
        fake_load_vad_model,
    )

    result = backend.load()

    assert result == "fake-model"
    assert calls == {"run_opts": {"device": "cpu"}}


def test_speechbrain_backend_normalizes_segments_and_frame_probabilities():
    backend = SpeechBrainFP32Backend()

    class FakeVADModel:
        @staticmethod
        def get_speech_segments(audio_file):
            assert audio_file == "clip.wav"
            return [(0.0, 1.5), (2.0, 2.25)]

        @staticmethod
        def get_speech_prob_file(audio_file):
            assert audio_file == "clip.wav"
            return [[0.1], [0.9]]

    assert backend.predict_segments(FakeVADModel(), "clip.wav") == [
        {"start": 0.0, "end": 1.5, "duration": 1.5},
        {"start": 2.0, "end": 2.25, "duration": 0.25},
    ]
    assert backend.predict_frame_probabilities(FakeVADModel(), "clip.wav") == [
        {"frame_index": 0, "speech_probability": 0.1},
        {"frame_index": 1, "speech_probability": 0.9},
    ]


def test_list_backend_names_includes_energy_zcr():
    assert "energy_zcr" in list_backend_names()


def test_get_backend_returns_energy_zcr_backend():
    from vad_baseline.backends.energy_zcr import EnergyZCRBackend

    backend = get_backend("energy_zcr")

    assert isinstance(backend, EnergyZCRBackend)
    assert backend.backend_name == "energy_zcr"
    assert backend.model_name == "classical/energy_zcr"


def test_energy_zcr_detects_obvious_voiced_region(tmp_path):
    from vad_baseline.backends.energy_zcr import EnergyZCRBackend

    audio_path = tmp_path / "tone.wav"
    _write_waveform(
        audio_path,
        [
            (0.2, 0.0, None),
            (0.4, 0.8, 220),
            (0.2, 0.0, None),
        ],
    )
    backend = EnergyZCRBackend()

    segments = backend.predict_segments(None, audio_path)

    assert len(segments) == 1
    assert 0.18 <= segments[0]["start"] <= 0.24
    assert 0.56 <= segments[0]["end"] <= 0.64
    assert segments[0]["duration"] > 0.3


def test_energy_zcr_merges_short_gap_between_voiced_regions(tmp_path):
    from vad_baseline.backends.energy_zcr import EnergyZCRBackend

    audio_path = tmp_path / "merged.wav"
    _write_waveform(
        audio_path,
        [
            (0.2, 0.0, None),
            (0.16, 0.8, 220),
            (0.04, 0.0, None),
            (0.18, 0.8, 220),
            (0.2, 0.0, None),
        ],
    )
    backend = EnergyZCRBackend(min_silence_sec=0.08)

    segments = backend.predict_segments(None, audio_path)

    assert len(segments) == 1
    assert 0.18 <= segments[0]["start"] <= 0.24
    assert 0.54 <= segments[0]["end"] <= 0.66


def test_energy_zcr_reports_zero_model_tensor_footprint():
    from vad_baseline.backends.energy_zcr import EnergyZCRBackend

    backend = EnergyZCRBackend()

    assert backend.load() is None
    assert backend.summarize_model_tensors(None) == {
        "model_parameter_count": 0,
        "model_parameter_bytes": 0,
        "model_buffer_bytes": 0,
        "model_total_tensor_bytes": 0,
        "model_parameter_mb": 0.0,
        "model_buffer_mb": 0.0,
        "model_total_tensor_mb": 0.0,
    }


def test_list_backend_names_includes_webrtc_vad():
    assert "webrtc_vad" in list_backend_names()


def test_get_backend_returns_webrtc_vad_backend():
    from vad_baseline.backends.webrtc_vad import WebRTCVADBackend

    backend = get_backend("webrtc_vad")

    assert isinstance(backend, WebRTCVADBackend)
    assert backend.backend_name == "webrtc_vad"
    assert backend.model_name == "classical/webrtc_vad"


def test_webrtc_vad_converts_frame_decisions_to_segments(
    tmp_path,
    monkeypatch,
):
    decisions = iter([False, False, True, True, False, False, False, False])

    class FakeVad:
        def __init__(self, aggressiveness):
            assert aggressiveness == 2

        @staticmethod
        def is_speech(frame_bytes, sample_rate):
            assert sample_rate == 16000
            return next(decisions)

    fake_module = types.ModuleType("webrtcvad")
    fake_module.Vad = FakeVad
    monkeypatch.setitem(sys.modules, "webrtcvad", fake_module)

    from vad_baseline.backends.webrtc_vad import WebRTCVADBackend

    audio_path = tmp_path / "webrtc.wav"
    _write_waveform(audio_path, [(0.16, 0.8, 220)])
    backend = WebRTCVADBackend(
        min_speech_sec=0.02,
        min_silence_sec=0.0,
        hangover_frames=0,
    )

    segments = backend.predict_segments(backend.load(), audio_path)

    assert segments == [
        {"start": 0.04, "end": 0.08, "duration": 0.04},
    ]


def test_webrtc_vad_reports_zero_model_tensor_footprint():
    from vad_baseline.backends.webrtc_vad import WebRTCVADBackend

    backend = WebRTCVADBackend()

    assert backend.summarize_model_tensors(None) == {
        "model_parameter_count": 0,
        "model_parameter_bytes": 0,
        "model_buffer_bytes": 0,
        "model_total_tensor_bytes": 0,
        "model_parameter_mb": 0.0,
        "model_buffer_mb": 0.0,
        "model_total_tensor_mb": 0.0,
    }


def test_list_backend_names_includes_speechbrain_dynamic_int8():
    assert "speechbrain_dynamic_int8" in list_backend_names()


def test_get_backend_returns_speechbrain_dynamic_int8_backend():
    from vad_baseline.backends.speechbrain_dynamic_int8 import (
        SpeechBrainDynamicINT8Backend,
    )

    backend = get_backend("speechbrain_dynamic_int8")

    assert isinstance(backend, SpeechBrainDynamicINT8Backend)
    assert backend.backend_name == "speechbrain_dynamic_int8"
    assert (
        backend.model_name
        == "speechbrain/vad-crdnn-libriparty-dynamic-int8"
    )
    assert backend.supports_frame_probabilities is True


def test_speechbrain_dynamic_int8_backend_load_quantizes_gru_and_linear(
    monkeypatch,
):
    calls = {}

    class FakeVADModel:
        def __init__(self):
            self.mods = "fp32-mods"

    def fake_load_vad_model(run_opts=None):
        calls["run_opts"] = run_opts
        return FakeVADModel()

    def fake_quantize_dynamic(mods, module_types, dtype=None):
        calls["mods"] = mods
        calls["module_types"] = module_types
        calls["dtype"] = dtype
        return "int8-mods"

    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_dynamic_int8.load_vad_model",
        fake_load_vad_model,
    )
    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_dynamic_int8.quantize_dynamic",
        fake_quantize_dynamic,
    )

    backend = get_backend(
        "speechbrain_dynamic_int8",
        run_opts={"device": "cpu"},
    )

    model = backend.load()

    assert calls == {
        "run_opts": {"device": "cpu"},
        "mods": "fp32-mods",
        "module_types": {torch.nn.GRU, torch.nn.Linear},
        "dtype": torch.qint8,
    }
    assert model.mods == "int8-mods"


def test_speechbrain_dynamic_int8_backend_load_adds_flatten_parameters_compat(
    monkeypatch,
):
    QuantizedGRU = type(
        "GRU",
        (torch.nn.Module,),
        {"__module__": "torch.ao.nn.quantized.dynamic.modules.rnn"},
    )

    class FakeQuantizedMods(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = QuantizedGRU()

    class FakeVADModel:
        def __init__(self):
            self.mods = "fp32-mods"

    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_dynamic_int8.load_vad_model",
        lambda run_opts=None: FakeVADModel(),
    )
    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_dynamic_int8.quantize_dynamic",
        lambda mods, module_types, dtype=None: FakeQuantizedMods(),
    )

    backend = get_backend("speechbrain_dynamic_int8")
    model = backend.load()

    assert hasattr(model.mods.rnn, "flatten_parameters")
    assert model.mods.rnn.flatten_parameters() is None


def test_list_backend_names_includes_speechbrain_onnx_runtime():
    assert "speechbrain_onnx_runtime" in list_backend_names()


def test_get_backend_returns_speechbrain_onnx_runtime_backend():
    from vad_baseline.backends.speechbrain_onnx_runtime import (
        SpeechBrainONNXRuntimeBackend,
    )

    backend = get_backend(
        "speechbrain_onnx_runtime",
        onnx_model_path="outputs/onnx_export/model.onnx",
    )

    assert isinstance(backend, SpeechBrainONNXRuntimeBackend)
    assert backend.backend_name == "speechbrain_onnx_runtime"
    assert backend.model_name == "speechbrain/vad-crdnn-libriparty-onnx-runtime"
    assert backend.supports_frame_probabilities is False
    assert backend.onnx_model_path == "outputs/onnx_export/model.onnx"


def test_speechbrain_onnx_runtime_backend_load_uses_runtime_loader(
    monkeypatch,
    tmp_path,
):
    calls = {}

    def fake_load_onnx_vad_runtime(onnx_model_path):
        calls["onnx_model_path"] = onnx_model_path
        return "fake-onnx-runtime"

    monkeypatch.setattr(
        "vad_baseline.backends.speechbrain_onnx_runtime.load_onnx_vad_runtime",
        fake_load_onnx_vad_runtime,
    )

    backend = get_backend(
        "speechbrain_onnx_runtime",
        onnx_model_path=tmp_path / "model.onnx",
    )

    assert backend.load() == "fake-onnx-runtime"
    assert calls == {"onnx_model_path": str(tmp_path / "model.onnx")}


def test_speechbrain_onnx_runtime_backend_reports_artifact_size(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"1234")
    backend = get_backend(
        "speechbrain_onnx_runtime",
        onnx_model_path=model_path,
    )

    assert backend.summarize_model_tensors(None) == {
        "model_parameter_count": 0,
        "model_parameter_bytes": 0,
        "model_buffer_bytes": 0,
        "model_total_tensor_bytes": 0,
        "model_parameter_mb": 0.0,
        "model_buffer_mb": 0.0,
        "model_total_tensor_mb": 0.0,
        "model_artifact_bytes": 4,
        "model_artifact_mb": 0.000004,
    }
