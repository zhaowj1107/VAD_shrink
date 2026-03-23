import json

import pytest

from vad_baseline.metrics import (
    compute_segment_metrics,
    load_annotation_segments,
    merge_speech_segments,
)


def test_load_annotation_segments_reads_and_merges_valid_json(tmp_path):
    annotation = tmp_path / "labels.json"
    annotation.write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 0.5},
                {"start": 0.4, "end": 0.8},
                {"start": 1.0, "end": 1.2},
            ]
        )
    )

    assert load_annotation_segments(annotation) == [
        {"start": 0.0, "end": 0.8, "duration": 0.8},
        {"start": 1.0, "end": 1.2, "duration": 0.2},
    ]


def test_load_annotation_segments_rejects_invalid_segment_values(tmp_path):
    annotation = tmp_path / "labels.json"
    annotation.write_text(json.dumps([{"start": 0.4, "end": 0.2}]))

    with pytest.raises(ValueError, match="end must be greater than start"):
        load_annotation_segments(annotation)


def test_merge_speech_segments_merges_adjacent_and_overlapping_ranges():
    assert merge_speech_segments(
        [
            {"start": 0.0, "end": 0.5},
            {"start": 0.5, "end": 0.8},
            {"start": 1.0, "end": 1.3},
            {"start": 1.2, "end": 1.5},
        ]
    ) == [
        {"start": 0.0, "end": 0.8, "duration": 0.8},
        {"start": 1.0, "end": 1.5, "duration": 0.5},
    ]


def test_compute_segment_metrics_reports_duration_overlap_on_fixed_grid():
    metrics = compute_segment_metrics(
        reference_segments=[{"start": 0.0, "end": 0.5}],
        predicted_segments=[{"start": 0.2, "end": 0.7}],
        time_resolution_sec=0.1,
    )

    assert metrics == {
        "reference_speech_sec": 0.5,
        "predicted_speech_sec": 0.5,
        "tp_sec": 0.3,
        "fp_sec": 0.2,
        "fn_sec": 0.2,
        "precision": 0.6,
        "recall": 0.6,
        "f1": 0.6,
        "false_alarm_rate": 0.4,
        "miss_rate": 0.4,
        "time_resolution_sec": 0.1,
    }


def test_compute_segment_metrics_handles_empty_predictions():
    metrics = compute_segment_metrics(
        reference_segments=[{"start": 0.0, "end": 0.3}],
        predicted_segments=[],
        time_resolution_sec=0.1,
    )

    assert metrics == {
        "reference_speech_sec": 0.3,
        "predicted_speech_sec": 0.0,
        "tp_sec": 0.0,
        "fp_sec": 0.0,
        "fn_sec": 0.3,
        "precision": None,
        "recall": 0.0,
        "f1": None,
        "false_alarm_rate": None,
        "miss_rate": 1.0,
        "time_resolution_sec": 0.1,
    }
