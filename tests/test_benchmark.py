import pytest

from vad_baseline.benchmark import build_benchmark_summary


def test_build_benchmark_summary_keeps_expected_fields():
    summary = build_benchmark_summary(
        model_name="speechbrain/vad-crdnn-libriparty",
        audio_duration_sec=5.0,
        inference_time_sec=0.25,
    )
    assert summary["model_name"] == "speechbrain/vad-crdnn-libriparty"
    assert summary["audio_duration_sec"] == 5.0
    assert summary["inference_time_sec"] == 0.25
    assert summary["rtf"] == 0.05


def test_build_benchmark_summary_rejects_zero_duration():
    with pytest.raises(ValueError, match="audio_duration_sec must be positive"):
        build_benchmark_summary(
            model_name="speechbrain/vad-crdnn-libriparty",
            audio_duration_sec=0.0,
            inference_time_sec=0.25,
        )
