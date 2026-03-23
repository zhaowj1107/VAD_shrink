import csv
import json
import shutil
from pathlib import Path

from vad_baseline.io_utils import write_json
from vad_baseline.metrics import merge_speech_segments

VALID_SUBSETS = {"train", "dev", "eval"}


def load_libriparty_session_segments(session_json_path):
    payload = json.loads(Path(session_json_path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("LibriParty session metadata must be an object")

    segments = []
    for utterances in payload.values():
        if not isinstance(utterances, list):
            continue
        for utterance in utterances:
            if not isinstance(utterance, dict):
                raise ValueError("LibriParty utterance must be an object")
            if "start" not in utterance or "stop" not in utterance:
                raise ValueError("LibriParty utterance must include start and stop")
            segments.append(
                {
                    "start": utterance["start"],
                    "end": utterance["stop"],
                }
            )

    return merge_speech_segments(segments)


def _subset_session_root(dataset_root, subset):
    dataset_root = Path(dataset_root)
    if subset not in VALID_SUBSETS:
        raise ValueError(f"unsupported subset: {subset}")

    subset_root = dataset_root / subset
    if not subset_root.is_dir():
        raise FileNotFoundError(subset_root)
    return subset_root


def _session_sort_key(session_dir):
    name = session_dir.name
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return name


def list_libriparty_subset_sessions(dataset_root, subset):
    subset_root = _subset_session_root(dataset_root, subset)
    sessions = []
    for session_dir in sorted(subset_root.glob("session_*"), key=_session_sort_key):
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name
        audio_path = (session_dir / f"{session_name}_mixture.wav").resolve()
        session_json_path = (session_dir / f"{session_name}.json").resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if not session_json_path.is_file():
            raise FileNotFoundError(session_json_path)

        sessions.append(
            {
                "id": f"{subset}_{session_name}",
                "subset": subset,
                "session_name": session_name,
                "audio_path": str(audio_path),
                "session_json_path": str(session_json_path),
            }
        )

    return sessions


def _resolve_subsets(subset):
    if subset == "all":
        return ["train", "dev", "eval"]
    if subset not in VALID_SUBSETS:
        raise ValueError(f"unsupported subset: {subset}")
    return [subset]


def _write_manifest(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "audio_path", "annotation_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_libriparty_manifest(
    dataset_root,
    output_dir,
    subset="dev",
    limit=None,
    overwrite=False,
):
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError("output_dir already exists")
        shutil.rmtree(output_dir)

    subset_names = _resolve_subsets(subset)
    all_sessions = []
    for subset_name in subset_names:
        all_sessions.extend(list_libriparty_subset_sessions(dataset_root, subset_name))

    num_sessions_found = len(all_sessions)
    selected_sessions = all_sessions[:limit] if limit is not None else all_sessions
    manifest_rows = []
    num_failed = 0

    for session in selected_sessions:
        try:
            segments = load_libriparty_session_segments(session["session_json_path"])
            annotation_path = (
                output_dir / "annotations" / f"{session['id']}.json"
            ).resolve()
            write_json(annotation_path, segments)
            manifest_rows.append(
                {
                    "id": session["id"],
                    "audio_path": session["audio_path"],
                    "annotation_path": str(annotation_path),
                }
            )
        except Exception:
            num_failed += 1

    _write_manifest(output_dir / "manifest.csv", manifest_rows)
    summary = {
        "subset": subset,
        "num_sessions_found": num_sessions_found,
        "num_generated": len(manifest_rows),
        "num_failed": num_failed,
        "num_skipped": num_sessions_found - len(selected_sessions),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
