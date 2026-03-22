import json

from vad_baseline.io_utils import write_frame_probs_csv, write_json


def test_write_json_creates_parent_dirs(tmp_path):
    output = tmp_path / "nested" / "segments.json"
    write_json(output, [{"start": 0.0, "end": 1.0, "duration": 1.0}])
    assert output.exists()
    assert json.loads(output.read_text()) == [
        {"start": 0.0, "end": 1.0, "duration": 1.0}
    ]


def test_write_frame_probs_csv_writes_header_and_rows(tmp_path):
    output = tmp_path / "nested" / "frame_probs.csv"
    write_frame_probs_csv(
        output,
        [
            {"frame_index": 0, "speech_probability": 0.1},
            {"frame_index": 1, "speech_probability": 0.9},
        ],
    )
    assert output.exists()
    assert output.read_text().splitlines() == [
        "frame_index,speech_probability",
        "0,0.1",
        "1,0.9",
    ]
