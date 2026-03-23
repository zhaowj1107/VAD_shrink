import csv
from pathlib import Path
from time import perf_counter

from vad_baseline.benchmark import build_benchmark_summary
from vad_baseline.inference import (
    get_frame_probabilities_for_file,
    get_wav_duration_sec,
    run_vad_on_file,
)
from vad_baseline.io_utils import write_frame_probs_csv, write_json, write_jsonl
from vad_baseline.metrics import compute_segment_metrics, load_annotation_segments
from vad_baseline.model import load_vad_model

REQUIRED_MANIFEST_COLUMNS = {"id", "audio_path"}
METRIC_FIELDS = [
    "reference_speech_sec",
    "predicted_speech_sec",
    "tp_sec",
    "fp_sec",
    "fn_sec",
    "precision",
    "recall",
    "f1",
    "false_alarm_rate",
    "miss_rate",
    "time_resolution_sec",
]


def _empty_metric_fields():
    return {field_name: None for field_name in METRIC_FIELDS}


def read_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        has_annotation_column = "annotation_path" in fieldnames
        missing_columns = REQUIRED_MANIFEST_COLUMNS - fieldnames
        if missing_columns:
            raise ValueError(
                f"manifest missing required columns: {sorted(missing_columns)}"
            )

        rows = []
        seen_ids = set()
        for row in reader:
            item_id = row["id"].strip()
            audio_path = row["audio_path"].strip()
            annotation_path = (row.get("annotation_path") or "").strip()
            if item_id in seen_ids:
                raise ValueError(f"duplicate id: {item_id}")

            resolved_audio_path = Path(audio_path)
            if not resolved_audio_path.is_absolute():
                resolved_audio_path = (
                    manifest_path.parent / resolved_audio_path
                ).resolve()

            resolved_annotation_path = None
            if annotation_path:
                resolved_annotation_path = Path(annotation_path)
                if not resolved_annotation_path.is_absolute():
                    resolved_annotation_path = (
                        manifest_path.parent / resolved_annotation_path
                    ).resolve()
                resolved_annotation_path = str(resolved_annotation_path)

            entry = {
                "id": item_id,
                "audio_path": str(resolved_audio_path),
            }
            if has_annotation_column:
                entry["annotation_path"] = resolved_annotation_path

            rows.append(entry)
            seen_ids.add(item_id)

    return rows


def _percentile_nearest_rank(values, percentile):
    sorted_values = sorted(values)
    index = max(
        0,
        min(
            len(sorted_values) - 1,
            int((percentile / 100) * len(sorted_values) + 0.9999999) - 1,
        ),
    )
    return sorted_values[index]


def _stable_mean(values):
    return round(sum(values) / len(values), 12)


def _stable_sum(values):
    return round(sum(values), 12)


def _mean_optional_field(rows, field_name):
    values = [row[field_name] for row in rows if row.get(field_name) is not None]
    if not values:
        return None
    return _stable_mean(values)


def process_manifest_entry(
    entry,
    vad_model,
    output_dir,
    save_frame_probs=False,
    timer=perf_counter,
):
    audio_path = Path(entry["audio_path"])
    item_id = entry["id"]
    annotation_path = entry.get("annotation_path")
    has_annotation = bool(annotation_path)
    segments_rel_path = Path("items") / item_id / "segments.json"
    frame_probs_rel_path = Path("items") / item_id / "frame_probs.csv"
    metrics_rel_path = Path("items") / item_id / "metrics.json"

    try:
        audio_duration_sec = get_wav_duration_sec(audio_path)
        started_at = timer()
        segments = run_vad_on_file(vad_model, audio_path)
        inference_time_sec = timer() - started_at

        write_json(Path(output_dir) / segments_rel_path, segments)

        metrics = _empty_metric_fields()
        scored = False
        metrics_path = None
        if has_annotation:
            reference_segments = load_annotation_segments(annotation_path)
            metrics = compute_segment_metrics(reference_segments, segments)
            write_json(Path(output_dir) / metrics_rel_path, metrics)
            scored = True
            metrics_path = metrics_rel_path.as_posix()

        if save_frame_probs:
            frame_probabilities = get_frame_probabilities_for_file(
                vad_model,
                audio_path,
            )
            write_frame_probs_csv(
                Path(output_dir) / frame_probs_rel_path,
                frame_probabilities,
            )

        benchmark = build_benchmark_summary(
            model_name="speechbrain/vad-crdnn-libriparty",
            audio_duration_sec=audio_duration_sec,
            inference_time_sec=inference_time_sec,
        )

        return {
            "id": item_id,
            "audio_path": str(audio_path),
            "annotation_path": annotation_path,
            "status": "success",
            "audio_duration_sec": audio_duration_sec,
            "inference_time_sec": benchmark["inference_time_sec"],
            "rtf": benchmark["rtf"],
            "num_segments": len(segments),
            "segments_path": segments_rel_path.as_posix(),
            "frame_probs_path": (
                frame_probs_rel_path.as_posix() if save_frame_probs else None
            ),
            "metrics_path": metrics_path,
            "has_annotation": has_annotation,
            "scored": scored,
            **metrics,
            "error": None,
        }
    except Exception as error:
        return {
            "id": item_id,
            "audio_path": str(audio_path),
            "annotation_path": annotation_path,
            "status": "failed",
            "audio_duration_sec": None,
            "inference_time_sec": None,
            "rtf": None,
            "num_segments": None,
            "segments_path": None,
            "frame_probs_path": None,
            "metrics_path": None,
            "has_annotation": has_annotation,
            "scored": False,
            **_empty_metric_fields(),
            "error": str(error),
        }


def summarize_results(results):
    successful = [row for row in results if row["status"] == "success"]
    scored = [row for row in successful if row.get("scored")]
    inference_times = [row["inference_time_sec"] for row in successful]
    rtfs = [row["rtf"] for row in successful]

    if successful:
        mean_inference_time_sec = _stable_mean(inference_times)
        mean_rtf = _stable_mean(rtfs)
        p50_inference_time_sec = _percentile_nearest_rank(inference_times, 50)
        p95_inference_time_sec = _percentile_nearest_rank(inference_times, 95)
    else:
        mean_inference_time_sec = None
        mean_rtf = None
        p50_inference_time_sec = None
        p95_inference_time_sec = None

    if scored:
        total_reference_speech_sec = _stable_sum(
            [row["reference_speech_sec"] for row in scored]
        )
        total_predicted_speech_sec = _stable_sum(
            [row["predicted_speech_sec"] for row in scored]
        )
        total_tp_sec = _stable_sum([row["tp_sec"] for row in scored])
        total_fp_sec = _stable_sum([row["fp_sec"] for row in scored])
        total_fn_sec = _stable_sum([row["fn_sec"] for row in scored])
        mean_precision = _mean_optional_field(scored, "precision")
        mean_recall = _mean_optional_field(scored, "recall")
        mean_f1 = _mean_optional_field(scored, "f1")
        mean_false_alarm_rate = _mean_optional_field(scored, "false_alarm_rate")
        mean_miss_rate = _mean_optional_field(scored, "miss_rate")
        scoring_time_resolution_sec = scored[0]["time_resolution_sec"]
    else:
        total_reference_speech_sec = None
        total_predicted_speech_sec = None
        total_tp_sec = None
        total_fp_sec = None
        total_fn_sec = None
        mean_precision = None
        mean_recall = None
        mean_f1 = None
        mean_false_alarm_rate = None
        mean_miss_rate = None
        scoring_time_resolution_sec = None

    return {
        "num_total": len(results),
        "num_success": len(successful),
        "num_failed": len(results) - len(successful),
        "num_scored": len(scored),
        "mean_inference_time_sec": mean_inference_time_sec,
        "mean_rtf": mean_rtf,
        "p50_inference_time_sec": p50_inference_time_sec,
        "p95_inference_time_sec": p95_inference_time_sec,
        "total_reference_speech_sec": total_reference_speech_sec,
        "total_predicted_speech_sec": total_predicted_speech_sec,
        "total_tp_sec": total_tp_sec,
        "total_fp_sec": total_fp_sec,
        "total_fn_sec": total_fn_sec,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_false_alarm_rate": mean_false_alarm_rate,
        "mean_miss_rate": mean_miss_rate,
        "scoring_time_resolution_sec": scoring_time_resolution_sec,
    }


def run_batch_evaluation(
    manifest_path,
    output_dir,
    save_frame_probs=False,
):
    entries = read_manifest(manifest_path)
    output_dir = Path(output_dir)
    vad_model = load_vad_model()

    results = [
        process_manifest_entry(
            entry,
            vad_model,
            output_dir,
            save_frame_probs=save_frame_probs,
        )
        for entry in entries
    ]
    summary = summarize_results(results)

    write_jsonl(output_dir / "per_file.jsonl", results)
    write_json(output_dir / "summary.json", summary)

    return summary
