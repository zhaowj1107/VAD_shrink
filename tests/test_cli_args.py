from pathlib import Path

import pytest

from scripts import run_inference
from scripts.run_inference import build_parser


def test_cli_requires_audio_path():
    parser = build_parser()
    args = parser.parse_args(["input.wav"])
    assert args.audio_path == "input.wav"


def test_main_rejects_missing_audio_before_loading_model(monkeypatch, tmp_path):
    model_loaded = False

    def fake_load_vad_model():
        nonlocal model_loaded
        model_loaded = True

    monkeypatch.setattr(run_inference, "load_vad_model", fake_load_vad_model)

    missing_audio = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError, match=str(missing_audio)):
        run_inference.main([str(missing_audio)])

    assert model_loaded is False


def test_main_rejects_invalid_wav_before_loading_model(monkeypatch, tmp_path):
    model_loaded = False

    def fake_load_vad_model():
        nonlocal model_loaded
        model_loaded = True

    audio_path = tmp_path / "invalid.wav"
    audio_path.write_bytes(b"not-a-real-wave")

    monkeypatch.setattr(run_inference, "load_vad_model", fake_load_vad_model)
    monkeypatch.setattr(
        run_inference,
        "get_wav_duration_sec",
        lambda _: (_ for _ in ()).throw(ValueError("invalid wav file")),
    )

    with pytest.raises(ValueError, match="invalid wav file"):
        run_inference.main([str(audio_path)])

    assert model_loaded is False


def test_main_writes_expected_artifacts_and_times_only_inference(
    monkeypatch,
    tmp_path,
):
    events = []
    output_dir = tmp_path / "out"
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"RIFF")

    def fake_load_vad_model():
        events.append("load")
        return "fake-model"

    def fake_run_vad_on_file(model, path):
        events.append("run")
        assert model == "fake-model"
        assert Path(path) == audio_path
        return [{"start": 0.0, "end": 1.0, "duration": 1.0}]

    def fake_get_frame_probabilities_for_file(model, path):
        events.append("frame_probs")
        assert model == "fake-model"
        assert Path(path) == audio_path
        return [{"frame_index": 0, "speech_probability": 0.5}]

    def fake_perf_counter():
        events.append("timer")
        return {1: 10.0, 2: 10.25}[events.count("timer")]

    written = []

    def fake_write_json(path, payload):
        written.append(("json", Path(path).name, payload))

    def fake_write_frame_probs_csv(path, rows):
        written.append(("csv", Path(path).name, rows))

    def fake_build_benchmark_summary(**kwargs):
        written.append(("benchmark", kwargs))
        return {"benchmark": True, **kwargs}

    monkeypatch.setattr(run_inference, "load_vad_model", fake_load_vad_model)
    monkeypatch.setattr(run_inference, "run_vad_on_file", fake_run_vad_on_file)
    monkeypatch.setattr(
        run_inference,
        "get_frame_probabilities_for_file",
        fake_get_frame_probabilities_for_file,
    )
    monkeypatch.setattr(run_inference, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(run_inference, "write_json", fake_write_json)
    monkeypatch.setattr(
        run_inference,
        "write_frame_probs_csv",
        fake_write_frame_probs_csv,
    )
    monkeypatch.setattr(
        run_inference,
        "build_benchmark_summary",
        fake_build_benchmark_summary,
    )
    monkeypatch.setattr(run_inference, "get_wav_duration_sec", lambda _: 2.0)

    exit_code = run_inference.main(
        [
            str(audio_path),
            "--output-dir",
            str(output_dir),
            "--save-frame-probs",
        ]
    )

    assert exit_code == 0
    assert events == ["load", "timer", "run", "timer", "frame_probs"]
    assert ("json", "segments.json", [{"start": 0.0, "end": 1.0, "duration": 1.0}]) in written
    assert (
        "csv",
        "frame_probs.csv",
        [{"frame_index": 0, "speech_probability": 0.5}],
    ) in written
    assert (
        "benchmark",
        {
            "model_name": "speechbrain/vad-crdnn-libriparty",
            "audio_duration_sec": 2.0,
            "inference_time_sec": 0.25,
        },
    ) in written
    assert (
        "json",
        "benchmark.json",
        {
            "benchmark": True,
            "model_name": "speechbrain/vad-crdnn-libriparty",
            "audio_duration_sec": 2.0,
            "inference_time_sec": 0.25,
        },
    ) in written
