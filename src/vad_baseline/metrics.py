import json
import math
from pathlib import Path


def _round_metric(value):
    return round(value, 12)


def _normalize_segment(segment):
    if not isinstance(segment, dict):
        raise ValueError("segment must be an object with start and end")
    if "start" not in segment or "end" not in segment:
        raise ValueError("segment must include start and end")

    start = float(segment["start"])
    end = float(segment["end"])
    if start < 0:
        raise ValueError("start must be greater than or equal to 0")
    if end <= start:
        raise ValueError("end must be greater than start")

    return {
        "start": start,
        "end": end,
        "duration": _round_metric(end - start),
    }


def merge_speech_segments(segments):
    normalized = sorted(
        (_normalize_segment(segment) for segment in segments),
        key=lambda item: (item["start"], item["end"]),
    )
    if not normalized:
        return []

    merged = [normalized[0].copy()]
    for segment in normalized[1:]:
        current = merged[-1]
        if segment["start"] <= current["end"]:
            current["end"] = max(current["end"], segment["end"])
            current["duration"] = _round_metric(current["end"] - current["start"])
            continue

        merged.append(segment.copy())

    return merged


def load_annotation_segments(annotation_path):
    payload = json.loads(Path(annotation_path).read_text())
    if not isinstance(payload, list):
        raise ValueError("annotation file must contain a list of segments")
    return merge_speech_segments(payload)


def _segments_to_activity(segments, time_resolution_sec):
    if time_resolution_sec <= 0:
        raise ValueError("time_resolution_sec must be positive")

    merged = merge_speech_segments(segments)
    if not merged:
        return []

    max_end = max(segment["end"] for segment in merged)
    num_steps = int(math.ceil(max_end / time_resolution_sec))
    activity = [False] * num_steps

    for step_index in range(num_steps):
        step_start = step_index * time_resolution_sec
        step_end = step_start + time_resolution_sec
        activity[step_index] = any(
            step_start < segment["end"] and step_end > segment["start"]
            for segment in merged
        )

    return activity


def _safe_divide(numerator, denominator):
    if denominator <= 0:
        return None
    return _round_metric(numerator / denominator)


def compute_segment_metrics(
    reference_segments,
    predicted_segments,
    time_resolution_sec=0.01,
):
    reference_activity = _segments_to_activity(
        reference_segments,
        time_resolution_sec,
    )
    predicted_activity = _segments_to_activity(
        predicted_segments,
        time_resolution_sec,
    )
    num_steps = max(len(reference_activity), len(predicted_activity))
    reference_activity.extend([False] * (num_steps - len(reference_activity)))
    predicted_activity.extend([False] * (num_steps - len(predicted_activity)))

    tp_steps = sum(
        1
        for reference_on, predicted_on in zip(reference_activity, predicted_activity)
        if reference_on and predicted_on
    )
    fp_steps = sum(
        1
        for reference_on, predicted_on in zip(reference_activity, predicted_activity)
        if predicted_on and not reference_on
    )
    fn_steps = sum(
        1
        for reference_on, predicted_on in zip(reference_activity, predicted_activity)
        if reference_on and not predicted_on
    )
    reference_steps = sum(reference_activity)
    predicted_steps = sum(predicted_activity)

    tp_sec = _round_metric(tp_steps * time_resolution_sec)
    fp_sec = _round_metric(fp_steps * time_resolution_sec)
    fn_sec = _round_metric(fn_steps * time_resolution_sec)
    reference_speech_sec = _round_metric(reference_steps * time_resolution_sec)
    predicted_speech_sec = _round_metric(predicted_steps * time_resolution_sec)

    precision = _safe_divide(tp_sec, tp_sec + fp_sec)
    recall = _safe_divide(tp_sec, tp_sec + fn_sec)
    false_alarm_rate = _safe_divide(fp_sec, tp_sec + fp_sec)
    miss_rate = _safe_divide(fn_sec, tp_sec + fn_sec)

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = _round_metric((2 * precision * recall) / (precision + recall))

    return {
        "reference_speech_sec": reference_speech_sec,
        "predicted_speech_sec": predicted_speech_sec,
        "tp_sec": tp_sec,
        "fp_sec": fp_sec,
        "fn_sec": fn_sec,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "time_resolution_sec": time_resolution_sec,
    }
