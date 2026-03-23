import json

import torch

from vad_baseline import profiling


class _FakeProfileModel:
    def __init__(self):
        self.mods = torch.nn.ModuleDict(
            {"linear": torch.nn.Linear(4, 2, bias=False)}
        )
        self.mods.register_buffer("scale", torch.ones(3, dtype=torch.float32))


def test_summarize_model_tensors_reports_parameter_and_buffer_sizes():
    summary = profiling.summarize_model_tensors(_FakeProfileModel())

    assert summary == {
        "model_parameter_count": 8,
        "model_parameter_bytes": 32,
        "model_buffer_bytes": 12,
        "model_total_tensor_bytes": 44,
        "model_parameter_mb": 0.000031,
        "model_buffer_mb": 0.000011,
        "model_total_tensor_mb": 0.000042,
    }


def test_parse_rss_mb_reads_linux_status_format():
    assert profiling.parse_rss_mb("Name:\tpython\nVmRSS:\t   2048 kB\n") == 2.0


def test_delta_cpu_times_computes_user_and_system_deltas():
    assert profiling.delta_cpu_times(
        {"cpu_user_sec": 1.0, "cpu_system_sec": 2.0},
        {"cpu_user_sec": 1.25, "cpu_system_sec": 2.5},
    ) == {"cpu_user_sec": 0.25, "cpu_system_sec": 0.5}


def test_profile_batch_manifest_writes_profile_and_batch_artifacts(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,a.wav\nutt2,b.wav\n")
    output_dir = tmp_path / "profile_out"
    calls = {"loads": 0, "processed": []}
    timer_values = iter([10.0, 10.2, 12.0, 14.5])
    rss_values = iter([100.0, 140.0, 150.0])
    peak_rss_values = iter([120.0, 165.0, 180.0])
    cpu_values = iter(
        [
            {"cpu_user_sec": 1.0, "cpu_system_sec": 2.0},
            {"cpu_user_sec": 1.3, "cpu_system_sec": 2.2},
            {"cpu_user_sec": 2.1, "cpu_system_sec": 2.7},
        ]
    )

    def fake_load_model():
        calls["loads"] += 1
        return _FakeProfileModel()

    def fake_process_manifest_entry(
        entry,
        vad_model,
        output_dir,
        save_frame_probs=False,
        timer=None,
    ):
        calls["processed"].append(
            (entry["id"], type(vad_model).__name__, str(output_dir), save_frame_probs)
        )
        return {
            "id": entry["id"],
            "audio_path": entry["audio_path"],
            "annotation_path": None,
            "status": "success",
            "audio_duration_sec": 2.0,
            "inference_time_sec": 0.5,
            "rtf": 0.25,
            "num_segments": 1,
            "segments_path": f"items/{entry['id']}/segments.json",
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

    profile = profiling.profile_batch_manifest(
        manifest_path=manifest,
        output_dir=output_dir,
        save_frame_probs=True,
        load_model_fn=fake_load_model,
        timer=lambda: next(timer_values),
        current_rss_reader=lambda: next(rss_values),
        peak_rss_reader=lambda: next(peak_rss_values),
        cpu_times_reader=lambda: next(cpu_values),
        process_manifest_entry_fn=fake_process_manifest_entry,
    )

    assert calls["loads"] == 1
    assert calls["processed"] == [
        ("utt1", "_FakeProfileModel", str(output_dir), True),
        ("utt2", "_FakeProfileModel", str(output_dir), True),
    ]
    assert profile == {
        "manifest_path": str(manifest.resolve()),
        "model_name": "speechbrain/vad-crdnn-libriparty",
        "num_entries": 2,
        "load_time_sec": 0.2,
        "run_wall_time_sec": 2.5,
        "total_wall_time_sec": 2.7,
        "rss_before_load_mb": 100.0,
        "rss_after_load_mb": 140.0,
        "rss_after_run_mb": 150.0,
        "peak_rss_before_load_mb": 120.0,
        "peak_rss_after_load_mb": 165.0,
        "peak_rss_after_run_mb": 180.0,
        "cpu_user_load_sec": 0.3,
        "cpu_system_load_sec": 0.2,
        "cpu_user_run_sec": 0.8,
        "cpu_system_run_sec": 0.5,
        "model_parameter_count": 8,
        "model_parameter_bytes": 32,
        "model_buffer_bytes": 12,
        "model_total_tensor_bytes": 44,
        "model_parameter_mb": 0.000031,
        "model_buffer_mb": 0.000011,
        "model_total_tensor_mb": 0.000042,
        "batch_summary": {
            "num_total": 2,
            "num_success": 2,
            "num_failed": 0,
            "num_scored": 0,
            "mean_inference_time_sec": 0.5,
            "mean_rtf": 0.25,
            "p50_inference_time_sec": 0.5,
            "p95_inference_time_sec": 0.5,
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
        },
    }
    assert [
        json.loads(line)
        for line in (output_dir / "per_file.jsonl").read_text().splitlines()
    ][0]["id"] == "utt1"
    assert json.loads((output_dir / "summary.json").read_text()) == profile["batch_summary"]
    assert json.loads((output_dir / "profile.json").read_text()) == profile


def test_profile_batch_manifest_uses_backend_and_records_backend_name(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,a.wav\n")
    output_dir = tmp_path / "profile_out"
    timer_values = iter([1.0, 1.1, 3.0, 3.4])
    rss_values = iter([100.0, 110.0, 115.0])
    peak_rss_values = iter([120.0, 122.0, 126.0])
    cpu_values = iter(
        [
            {"cpu_user_sec": 1.0, "cpu_system_sec": 2.0},
            {"cpu_user_sec": 1.2, "cpu_system_sec": 2.1},
            {"cpu_user_sec": 1.5, "cpu_system_sec": 2.4},
        ]
    )
    calls = {"loads": 0, "processed": []}

    class FakeBackend:
        backend_name = "fake_backend"
        model_name = "fake/model"

        @staticmethod
        def load():
            calls["loads"] += 1
            return "fake-model"

        @staticmethod
        def summarize_model_tensors(vad_model):
            assert vad_model == "fake-model"
            return {
                "model_parameter_count": 0,
                "model_parameter_bytes": 0,
                "model_buffer_bytes": 0,
                "model_total_tensor_bytes": 0,
                "model_parameter_mb": 0.0,
                "model_buffer_mb": 0.0,
                "model_total_tensor_mb": 0.0,
            }

    def fake_process_manifest_entry(
        entry,
        vad_model,
        output_dir,
        save_frame_probs=False,
        timer=None,
        backend=None,
    ):
        calls["processed"].append(
            (
                entry["id"],
                vad_model,
                str(output_dir),
                save_frame_probs,
                backend.backend_name,
            )
        )
        return {
            "id": entry["id"],
            "audio_path": entry["audio_path"],
            "annotation_path": None,
            "backend_name": backend.backend_name,
            "model_name": backend.model_name,
            "status": "success",
            "audio_duration_sec": 2.0,
            "inference_time_sec": 0.25,
            "rtf": 0.125,
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

    profile = profiling.profile_batch_manifest(
        manifest_path=manifest,
        output_dir=output_dir,
        backend=FakeBackend(),
        timer=lambda: next(timer_values),
        current_rss_reader=lambda: next(rss_values),
        peak_rss_reader=lambda: next(peak_rss_values),
        cpu_times_reader=lambda: next(cpu_values),
        process_manifest_entry_fn=fake_process_manifest_entry,
    )

    assert calls["loads"] == 1
    assert calls["processed"] == [
        ("utt1", "fake-model", str(output_dir), False, "fake_backend")
    ]
    assert profile["backend_name"] == "fake_backend"
    assert profile["model_name"] == "fake/model"
    assert profile["batch_summary"]["backend_name"] == "fake_backend"
    assert "memory_stages" not in profile


def test_profile_batch_manifest_records_memory_stages_for_speechbrain_backend(
    tmp_path,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,a.wav\nutt2,b.wav\n")
    output_dir = tmp_path / "profile_out"
    timer_values = iter([10.0, 10.1, 12.0, 14.0])
    rss_values = iter([100.0, 120.0, 125.0, 150.0, 155.0, 180.0])
    peak_rss_values = iter([110.0, 135.0, 140.0, 165.0, 170.0, 195.0])
    cpu_values = iter(
        [
            {"cpu_user_sec": 1.0, "cpu_system_sec": 2.0},
            {"cpu_user_sec": 1.2, "cpu_system_sec": 2.1},
            {"cpu_user_sec": 1.25, "cpu_system_sec": 2.15},
            {"cpu_user_sec": 1.5, "cpu_system_sec": 2.25},
            {"cpu_user_sec": 1.55, "cpu_system_sec": 2.3},
            {"cpu_user_sec": 2.0, "cpu_system_sec": 2.8},
        ]
    )
    calls = {"loads": 0, "processed": []}

    class FakeSpeechBrainBackend:
        backend_name = "speechbrain_fp32"
        model_name = "speechbrain/vad-crdnn-libriparty"

        @staticmethod
        def load():
            calls["loads"] += 1
            return _FakeProfileModel()

        @staticmethod
        def summarize_model_tensors(vad_model):
            return profiling.summarize_model_tensors(vad_model)

    def fake_process_manifest_entry(
        entry,
        vad_model,
        output_dir,
        save_frame_probs=False,
        timer=None,
        backend=None,
        stage_callbacks=None,
    ):
        calls["processed"].append(
            (
                entry["id"],
                type(vad_model).__name__,
                str(output_dir),
                save_frame_probs,
                backend.backend_name,
                bool(stage_callbacks),
            )
        )
        if stage_callbacks:
            stage_callbacks["after_metadata"]()
            stage_callbacks["after_inference"]()
            stage_callbacks["after_scoring"]()
        return {
            "id": entry["id"],
            "audio_path": entry["audio_path"],
            "annotation_path": None,
            "backend_name": backend.backend_name,
            "model_name": backend.model_name,
            "status": "success",
            "audio_duration_sec": 2.0,
            "inference_time_sec": 0.5,
            "rtf": 0.25,
            "num_segments": 1,
            "segments_path": f"items/{entry['id']}/segments.json",
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

    profile = profiling.profile_batch_manifest(
        manifest_path=manifest,
        output_dir=output_dir,
        backend=FakeSpeechBrainBackend(),
        timer=lambda: next(timer_values),
        current_rss_reader=lambda: next(rss_values),
        peak_rss_reader=lambda: next(peak_rss_values),
        cpu_times_reader=lambda: next(cpu_values),
        process_manifest_entry_fn=fake_process_manifest_entry,
    )

    assert calls["loads"] == 1
    assert calls["processed"] == [
        (
            "utt1",
            "_FakeProfileModel",
            str(output_dir),
            False,
            "speechbrain_fp32",
            True,
        ),
        (
            "utt2",
            "_FakeProfileModel",
            str(output_dir),
            False,
            "speechbrain_fp32",
            False,
        ),
    ]
    assert profile["num_entries"] == 2
    assert profile["memory_stages"] == [
        {
            "stage_name": "process_start",
            "rss_mb": 100.0,
            "peak_rss_mb": 110.0,
            "cpu_user_sec": 0.0,
            "cpu_system_sec": 0.0,
            "delta_rss_from_previous_mb": 0.0,
            "delta_rss_from_start_mb": 0.0,
        },
        {
            "stage_name": "after_backend_load",
            "rss_mb": 120.0,
            "peak_rss_mb": 135.0,
            "cpu_user_sec": 0.2,
            "cpu_system_sec": 0.1,
            "delta_rss_from_previous_mb": 20.0,
            "delta_rss_from_start_mb": 20.0,
        },
        {
            "stage_name": "after_first_entry_metadata",
            "rss_mb": 125.0,
            "peak_rss_mb": 140.0,
            "cpu_user_sec": 0.25,
            "cpu_system_sec": 0.15,
            "delta_rss_from_previous_mb": 5.0,
            "delta_rss_from_start_mb": 25.0,
        },
        {
            "stage_name": "after_first_entry_inference",
            "rss_mb": 150.0,
            "peak_rss_mb": 165.0,
            "cpu_user_sec": 0.5,
            "cpu_system_sec": 0.25,
            "delta_rss_from_previous_mb": 25.0,
            "delta_rss_from_start_mb": 50.0,
        },
        {
            "stage_name": "after_first_entry_scoring",
            "rss_mb": 155.0,
            "peak_rss_mb": 170.0,
            "cpu_user_sec": 0.55,
            "cpu_system_sec": 0.3,
            "delta_rss_from_previous_mb": 5.0,
            "delta_rss_from_start_mb": 55.0,
        },
        {
            "stage_name": "after_full_run",
            "rss_mb": 180.0,
            "peak_rss_mb": 195.0,
            "cpu_user_sec": 1.0,
            "cpu_system_sec": 0.8,
            "delta_rss_from_previous_mb": 25.0,
            "delta_rss_from_start_mb": 80.0,
        },
    ]
