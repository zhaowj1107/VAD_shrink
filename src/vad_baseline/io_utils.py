import csv
import json
from pathlib import Path


def write_json(path, payload) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def write_jsonl(path, rows) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else "")
    )


def write_frame_probs_csv(path, rows) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame_index", "speech_probability"],
        )
        writer.writeheader()
        writer.writerows(rows)
