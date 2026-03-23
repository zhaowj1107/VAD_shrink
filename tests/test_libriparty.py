import json
import wave
from pathlib import Path

import pytest

from vad_baseline.libriparty import (
    generate_libriparty_manifest,
    list_libriparty_subset_sessions,
    load_libriparty_session_segments,
)


def _write_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 16000):
    num_frames = int(duration_sec * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def _write_session(root: Path, subset: str, session_name: str, payload):
    session_dir = root / subset / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(session_dir / f"{session_name}_mixture.wav")
    (session_dir / f"{session_name}.json").write_text(json.dumps(payload))
    return session_dir


def test_load_libriparty_session_segments_merges_one_speaker(tmp_path):
    session_json = tmp_path / "session_0.json"
    session_json.write_text(
        json.dumps(
            {
                "6313": [
                    {"start": 0.32, "stop": 3.755},
                    {"start": 6.534, "stop": 8.979},
                ]
            }
        )
    )

    assert load_libriparty_session_segments(session_json) == [
        {"start": 0.32, "end": 3.755, "duration": 3.435},
        {"start": 6.534, "end": 8.979, "duration": 2.445},
    ]


def test_load_libriparty_session_segments_merges_multiple_speakers(tmp_path):
    session_json = tmp_path / "session_1.json"
    session_json.write_text(
        json.dumps(
            {
                "1455": [
                    {"start": 0.0, "stop": 1.0},
                    {"start": 3.0, "stop": 4.0},
                ],
                "163": [
                    {"start": 0.8, "stop": 1.2},
                    {"start": 4.0, "stop": 4.5},
                ],
            }
        )
    )

    assert load_libriparty_session_segments(session_json) == [
        {"start": 0.0, "end": 1.2, "duration": 1.2},
        {"start": 3.0, "end": 4.5, "duration": 1.5},
    ]


def test_load_libriparty_session_segments_rejects_invalid_intervals(tmp_path):
    session_json = tmp_path / "session_bad.json"
    session_json.write_text(
        json.dumps({"1455": [{"start": 1.0, "stop": 0.5}]})
    )

    with pytest.raises(ValueError, match="end must be greater than start"):
        load_libriparty_session_segments(session_json)


def test_list_libriparty_subset_sessions_sorts_by_numeric_session_id(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_session(
        dataset_root,
        "dev",
        "session_10",
        {"1455": [{"start": 0.0, "stop": 0.5}]},
    )
    _write_session(
        dataset_root,
        "dev",
        "session_2",
        {"163": [{"start": 1.0, "stop": 1.5}]},
    )

    sessions = list_libriparty_subset_sessions(dataset_root, "dev")

    assert sessions == [
        {
            "id": "dev_session_2",
            "subset": "dev",
            "session_name": "session_2",
            "audio_path": str(
                (dataset_root / "dev" / "session_2" / "session_2_mixture.wav").resolve()
            ),
            "session_json_path": str(
                (dataset_root / "dev" / "session_2" / "session_2.json").resolve()
            ),
        },
        {
            "id": "dev_session_10",
            "subset": "dev",
            "session_name": "session_10",
            "audio_path": str(
                (dataset_root / "dev" / "session_10" / "session_10_mixture.wav").resolve()
            ),
            "session_json_path": str(
                (dataset_root / "dev" / "session_10" / "session_10.json").resolve()
            ),
        },
    ]


def test_list_libriparty_subset_sessions_requires_expected_session_files(tmp_path):
    dataset_root = tmp_path / "dataset"
    session_dir = dataset_root / "dev" / "session_0"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session_0.json").write_text(
        json.dumps({"1455": [{"start": 0.0, "stop": 0.5}]})
    )

    with pytest.raises(FileNotFoundError, match="session_0_mixture.wav"):
        list_libriparty_subset_sessions(dataset_root, "dev")


def test_generate_libriparty_manifest_writes_outputs_and_honors_limit(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_session(
        dataset_root,
        "dev",
        "session_0",
        {"1455": [{"start": 0.0, "stop": 1.0}]},
    )
    _write_session(
        dataset_root,
        "dev",
        "session_1",
        {"163": [{"start": 2.0, "stop": 3.0}]},
    )
    output_dir = tmp_path / "generated"

    summary = generate_libriparty_manifest(
        dataset_root=dataset_root,
        output_dir=output_dir,
        subset="dev",
        limit=1,
    )

    assert summary == {
        "subset": "dev",
        "num_sessions_found": 2,
        "num_generated": 1,
        "num_failed": 0,
        "num_skipped": 1,
    }
    assert (output_dir / "manifest.csv").read_text().splitlines() == [
        "id,audio_path,annotation_path",
        (
            "dev_session_0,"
            f"{(dataset_root / 'dev' / 'session_0' / 'session_0_mixture.wav').resolve()},"
            f"{(output_dir / 'annotations' / 'dev_session_0.json').resolve()}"
        ),
    ]
    assert json.loads(
        (output_dir / "annotations" / "dev_session_0.json").read_text()
    ) == [{"start": 0.0, "end": 1.0, "duration": 1.0}]
    assert json.loads((output_dir / "summary.json").read_text()) == summary


def test_generate_libriparty_manifest_refuses_existing_output_without_overwrite(
    tmp_path,
):
    dataset_root = tmp_path / "dataset"
    _write_session(
        dataset_root,
        "dev",
        "session_0",
        {"1455": [{"start": 0.0, "stop": 1.0}]},
    )
    output_dir = tmp_path / "generated"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="output_dir already exists"):
        generate_libriparty_manifest(
            dataset_root=dataset_root,
            output_dir=output_dir,
            subset="dev",
        )
