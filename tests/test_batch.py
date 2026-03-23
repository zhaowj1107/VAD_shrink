import json
import wave
from pathlib import Path

import pytest

from vad_baseline import batch


def _write_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 16000):
    num_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def test_read_manifest_reads_valid_rows(tmp_path):
    audio = tmp_path / "a.wav"
    _write_wav(audio)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,a.wav\n")

    assert batch.read_manifest(manifest) == [
        {"id": "utt1", "audio_path": str(audio.resolve())}
    ]


def test_read_manifest_reads_optional_annotation_paths(tmp_path):
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    labels = tmp_path / "labels.json"
    _write_wav(audio_a)
    _write_wav(audio_b)
    labels.write_text('[{"start": 0.0, "end": 0.1}]')
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "id,audio_path,annotation_path\n"
        "utt1,a.wav,labels.json\n"
        "utt2,b.wav,\n"
    )

    assert batch.read_manifest(manifest) == [
        {
            "id": "utt1",
            "audio_path": str(audio_a.resolve()),
            "annotation_path": str(labels.resolve()),
        },
        {
            "id": "utt2",
            "audio_path": str(audio_b.resolve()),
            "annotation_path": None,
        },
    ]


def test_read_manifest_requires_expected_columns(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,path\nutt1,a.wav\n")

    with pytest.raises(ValueError, match="required columns"):
        batch.read_manifest(manifest)


def test_read_manifest_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,a.wav\nutt1,b.wav\n")

    with pytest.raises(ValueError, match="duplicate id"):
        batch.read_manifest(manifest)


def test_process_manifest_entry_writes_item_artifacts(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    _write_wav(audio)
    entry = {"id": "utt1", "audio_path": str(audio)}
    output_dir = tmp_path / "batch_out"
    timer_values = iter([10.0, 10.25])

    monkeypatch.setattr(
        batch,
        "get_wav_duration_sec",
        lambda path: 2.0,
    )
    monkeypatch.setattr(
        batch,
        "run_vad_on_file",
        lambda model, path: [{"start": 0.0, "end": 1.0, "duration": 1.0}],
    )
    monkeypatch.setattr(
        batch,
        "get_frame_probabilities_for_file",
        lambda model, path: [{"frame_index": 0, "speech_probability": 0.5}],
    )

    result = batch.process_manifest_entry(
        entry,
        vad_model="fake-model",
        output_dir=output_dir,
        save_frame_probs=True,
        timer=lambda: next(timer_values),
    )

    assert result == {
        "id": "utt1",
        "audio_path": str(audio),
        "annotation_path": None,
        "status": "success",
        "audio_duration_sec": 2.0,
        "inference_time_sec": 0.25,
        "rtf": 0.125,
        "num_segments": 1,
        "segments_path": "items/utt1/segments.json",
        "frame_probs_path": "items/utt1/frame_probs.csv",
        "metrics_path": None,
        "has_annotation": False,
        "scored": False,
        "reference_speech_sec": None,
        "predicted_speech_sec": None,
        "tp_sec": None,
        "fp_sec": None,
        "fn_sec": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "false_alarm_rate": None,
        "miss_rate": None,
        "time_resolution_sec": None,
        "error": None,
    }
    assert json.loads(
        (output_dir / "items" / "utt1" / "segments.json").read_text()
    ) == [{"start": 0.0, "end": 1.0, "duration": 1.0}]
    assert (
        output_dir / "items" / "utt1" / "frame_probs.csv"
    ).read_text().splitlines() == [
        "frame_index,speech_probability",
        "0,0.5",
    ]


def test_process_manifest_entry_writes_item_metrics_when_annotation_present(
    tmp_path,
    monkeypatch,
):
    audio = tmp_path / "a.wav"
    annotation = tmp_path / "labels.json"
    _write_wav(audio)
    annotation.write_text('[{"start": 0.0, "end": 0.5}]')
    output_dir = tmp_path / "batch_out"
    timer_values = iter([5.0, 5.3])
    metrics = {
        "reference_speech_sec": 0.5,
        "predicted_speech_sec": 0.6,
        "tp_sec": 0.4,
        "fp_sec": 0.2,
        "fn_sec": 0.1,
        "precision": 0.666666666667,
        "recall": 0.8,
        "f1": 0.727272727273,
        "false_alarm_rate": 0.333333333333,
        "miss_rate": 0.2,
        "time_resolution_sec": 0.01,
    }

    monkeypatch.setattr(batch, "get_wav_duration_sec", lambda path: 2.0)
    monkeypatch.setattr(
        batch,
        "run_vad_on_file",
        lambda model, path: [{"start": 0.0, "end": 0.6, "duration": 0.6}],
    )
    monkeypatch.setattr(
        batch,
        "load_annotation_segments",
        lambda path: [{"start": 0.0, "end": 0.5, "duration": 0.5}],
    )
    monkeypatch.setattr(
        batch,
        "compute_segment_metrics",
        lambda reference_segments, predicted_segments: metrics,
    )

    result = batch.process_manifest_entry(
        {
            "id": "utt1",
            "audio_path": str(audio),
            "annotation_path": str(annotation),
        },
        vad_model="fake-model",
        output_dir=output_dir,
        timer=lambda: next(timer_values),
    )

    assert result == {
        "id": "utt1",
        "audio_path": str(audio),
        "annotation_path": str(annotation),
        "status": "success",
        "audio_duration_sec": 2.0,
        "inference_time_sec": 0.3,
        "rtf": 0.15,
        "num_segments": 1,
        "segments_path": "items/utt1/segments.json",
        "frame_probs_path": None,
        "metrics_path": "items/utt1/metrics.json",
        "has_annotation": True,
        "scored": True,
        "reference_speech_sec": 0.5,
        "predicted_speech_sec": 0.6,
        "tp_sec": 0.4,
        "fp_sec": 0.2,
        "fn_sec": 0.1,
        "precision": 0.666666666667,
        "recall": 0.8,
        "f1": 0.727272727273,
        "false_alarm_rate": 0.333333333333,
        "miss_rate": 0.2,
        "time_resolution_sec": 0.01,
        "error": None,
    }
    assert json.loads(
        (output_dir / "items" / "utt1" / "metrics.json").read_text()
    ) == metrics


def test_process_manifest_entry_records_failures_without_raising(
    tmp_path,
    monkeypatch,
):
    missing_audio = tmp_path / "missing.wav"
    called = {"ran": False}

    def fake_run(model, path):
        called["ran"] = True
        return []

    monkeypatch.setattr(batch, "run_vad_on_file", fake_run)

    result = batch.process_manifest_entry(
        {"id": "bad", "audio_path": str(missing_audio)},
        vad_model="fake-model",
        output_dir=tmp_path / "batch_out",
    )

    assert called["ran"] is False
    assert result["status"] == "failed"
    assert result["error"]
    assert result["annotation_path"] is None
    assert result["has_annotation"] is False
    assert result["scored"] is False
    assert result["segments_path"] is None
    assert result["frame_probs_path"] is None
    assert result["metrics_path"] is None


def test_summarize_results_aggregates_successes_and_failures():
    summary = batch.summarize_results(
        [
            {
                "status": "success",
                "inference_time_sec": 1.0,
                "rtf": 0.1,
                "scored": False,
            },
            {
                "status": "failed",
                "inference_time_sec": None,
                "rtf": None,
                "scored": False,
            },
            {
                "status": "success",
                "inference_time_sec": 2.0,
                "rtf": 0.2,
                "scored": False,
            },
            {
                "status": "success",
                "inference_time_sec": 3.0,
                "rtf": 0.3,
                "scored": False,
            },
        ]
    )

    assert summary == {
        "num_total": 4,
        "num_success": 3,
        "num_failed": 1,
        "num_scored": 0,
        "mean_inference_time_sec": 2.0,
        "mean_rtf": 0.2,
        "p50_inference_time_sec": 2.0,
        "p95_inference_time_sec": 3.0,
        "total_reference_speech_sec": None,
        "total_predicted_speech_sec": None,
        "total_tp_sec": None,
        "total_fp_sec": None,
        "total_fn_sec": None,
        "mean_precision": None,
        "mean_recall": None,
        "mean_f1": None,
        "mean_false_alarm_rate": None,
        "mean_miss_rate": None,
        "scoring_time_resolution_sec": None,
    }


def test_summarize_results_aggregates_scored_metrics():
    summary = batch.summarize_results(
        [
            {
                "status": "success",
                "inference_time_sec": 1.0,
                "rtf": 0.1,
                "scored": True,
                "reference_speech_sec": 1.0,
                "predicted_speech_sec": 1.1,
                "tp_sec": 0.8,
                "fp_sec": 0.3,
                "fn_sec": 0.2,
                "precision": 0.727272727273,
                "recall": 0.8,
                "f1": 0.761904761905,
                "false_alarm_rate": 0.272727272727,
                "miss_rate": 0.2,
                "time_resolution_sec": 0.01,
            },
            {
                "status": "success",
                "inference_time_sec": 2.0,
                "rtf": 0.2,
                "scored": True,
                "reference_speech_sec": 0.5,
                "predicted_speech_sec": 0.4,
                "tp_sec": 0.3,
                "fp_sec": 0.1,
                "fn_sec": 0.2,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 0.666666666667,
                "false_alarm_rate": 0.25,
                "miss_rate": 0.4,
                "time_resolution_sec": 0.01,
            },
            {
                "status": "success",
                "inference_time_sec": 3.0,
                "rtf": 0.3,
                "scored": False,
            },
        ]
    )

    assert summary == {
        "num_total": 3,
        "num_success": 3,
        "num_failed": 0,
        "num_scored": 2,
        "mean_inference_time_sec": 2.0,
        "mean_rtf": 0.2,
        "p50_inference_time_sec": 2.0,
        "p95_inference_time_sec": 3.0,
        "total_reference_speech_sec": 1.5,
        "total_predicted_speech_sec": 1.5,
        "total_tp_sec": 1.1,
        "total_fp_sec": 0.4,
        "total_fn_sec": 0.4,
        "mean_precision": 0.738636363636,
        "mean_recall": 0.7,
        "mean_f1": 0.714285714286,
        "mean_false_alarm_rate": 0.261363636364,
        "mean_miss_rate": 0.3,
        "scoring_time_resolution_sec": 0.01,
    }


def test_run_batch_evaluation_loads_model_once_and_writes_outputs(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "id,audio_path\nutt1,a.wav\nutt2,b.wav\n",
    )
    output_dir = tmp_path / "batch_out"
    calls = {"loads": 0, "processed": []}

    def fake_load_vad_model():
        calls["loads"] += 1
        return "fake-model"

    def fake_process_manifest_entry(
        entry,
        vad_model,
        output_dir,
        save_frame_probs=False,
        timer=None,
    ):
        calls["processed"].append(
            (entry["id"], vad_model, str(output_dir), save_frame_probs)
        )
        if entry["id"] == "utt1":
            return {
                "id": "utt1",
                "audio_path": entry["audio_path"],
                "annotation_path": None,
                "status": "success",
                "audio_duration_sec": 2.0,
                "inference_time_sec": 0.5,
                "rtf": 0.25,
                "num_segments": 1,
                "segments_path": "items/utt1/segments.json",
                "frame_probs_path": None,
                "metrics_path": None,
                "has_annotation": False,
                "scored": False,
                "reference_speech_sec": None,
                "predicted_speech_sec": None,
                "tp_sec": None,
                "fp_sec": None,
                "fn_sec": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "false_alarm_rate": None,
                "miss_rate": None,
                "time_resolution_sec": None,
                "error": None,
            }

        return {
            "id": "utt2",
            "audio_path": entry["audio_path"],
            "annotation_path": None,
            "status": "failed",
            "audio_duration_sec": None,
            "inference_time_sec": None,
            "rtf": None,
            "num_segments": None,
            "segments_path": None,
            "frame_probs_path": None,
            "metrics_path": None,
            "has_annotation": False,
            "scored": False,
            "reference_speech_sec": None,
            "predicted_speech_sec": None,
            "tp_sec": None,
            "fp_sec": None,
            "fn_sec": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_alarm_rate": None,
            "miss_rate": None,
            "time_resolution_sec": None,
            "error": "bad wav",
        }

    monkeypatch.setattr(batch, "load_vad_model", fake_load_vad_model)
    monkeypatch.setattr(
        batch,
        "process_manifest_entry",
        fake_process_manifest_entry,
    )

    summary = batch.run_batch_evaluation(
        manifest,
        output_dir,
        save_frame_probs=True,
    )

    assert calls["loads"] == 1
    assert calls["processed"] == [
        ("utt1", "fake-model", str(output_dir), True),
        ("utt2", "fake-model", str(output_dir), True),
    ]
    assert summary["num_total"] == 2
    assert summary["num_success"] == 1
    assert summary["num_failed"] == 1

    rows = [
        json.loads(line)
        for line in (output_dir / "per_file.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in rows] == ["utt1", "utt2"]
    assert json.loads((output_dir / "summary.json").read_text()) == summary
