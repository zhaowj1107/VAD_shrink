# VAD Baseline Research Repo

This repository builds a minimal Linux baseline for offline voice activity detection using SpeechBrain's pretrained `vad-crdnn-libriparty` model. The repository metadata targets a CPU-oriented PyTorch baseline for Linux simulation.

Example environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate a Python environment that satisfies `requirements.txt`, then run:

```bash
PYTHONPATH=src python scripts/run_inference.py path/to/input.wav --output-dir outputs/run1
```

Optional frame-level probabilities:

```bash
PYTHONPATH=src python scripts/run_inference.py path/to/input.wav --output-dir outputs/run1 --save-frame-probs
```

Expected outputs:

- `outputs/run1/segments.json`
- `outputs/run1/benchmark.json`
- `outputs/run1/frame_probs.csv` when `--save-frame-probs` is enabled

Downloaded LibriParty data is currently stored under:

- `data/external/LibriParty/dataset`

Generate a LibriParty-specific manifest plus repository-native annotations:

```bash
PYTHONPATH=src python scripts/generate_libriparty_manifest.py \
  --dataset-root data/external/LibriParty/dataset \
  --subset dev \
  --limit 2 \
  --output-dir outputs/libriparty_dev_manifest
```

Generated outputs:

- `outputs/libriparty_dev_manifest/manifest.csv`
- `outputs/libriparty_dev_manifest/annotations/`
- `outputs/libriparty_dev_manifest/summary.json`

The generated `manifest.csv` feeds directly into the batch runner.

Batch evaluation from a CSV manifest:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/batch_run
```

Manifest format:

```csv
id,audio_path,annotation_path
utt1,samples/a.wav,labels/a.json
utt2,samples/b.wav,
```

Notes:

- `id` and `audio_path` are required
- `annotation_path` is optional
- relative `audio_path` and `annotation_path` values are resolved relative to the manifest file

Annotation format:

```json
[
  {"start": 0.20, "end": 1.10},
  {"start": 1.80, "end": 2.40}
]
```

Optional batch frame-level probabilities:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/batch_run --save-frame-probs
```

Expected batch outputs:

- `outputs/batch_run/per_file.jsonl`
- `outputs/batch_run/summary.json`
- `outputs/batch_run/items/<id>/segments.json`
- `outputs/batch_run/items/<id>/metrics.json` when `annotation_path` is provided
- `outputs/batch_run/items/<id>/frame_probs.csv` when `--save-frame-probs` is enabled

When annotations are present, `per_file.jsonl` includes segment-overlap quality fields such as `precision`, `recall`, `f1`, `false_alarm_rate`, and `miss_rate`. `summary.json` also includes aggregate scoring totals and mean quality metrics across scored files.

The `samples/` directory is reserved for small local smoke-test audio files and is kept in the repository with a placeholder marker.
