# VAD Baseline Research Repo

This repository builds a minimal Linux baseline for offline voice activity detection using SpeechBrain's pretrained `vad-crdnn-libriparty` model. The repository metadata targets a CPU-oriented PyTorch baseline for Linux simulation.

Currently supported batch/profiling backends:

- `speechbrain_fp32`
- `speechbrain_dynamic_int8`
- `speechbrain_static_int8`
- `speechbrain_onnx_runtime`
- `energy_zcr`
- `webrtc_vad`

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

Bootstrap the LibriParty dev dataset (downloads ~1 GB the first time, builds 10 sessions):

```bash
PYTHONPATH=src python scripts/bootstrap_libriparty_dataset.py --dev-sessions 10
```

This script downloads LibriSpeech dev-clean and RIRS_NOISES (if not already present under
`data/external/LibriParty/downloads/`) and generates synthetic multi-speaker mixtures under
`data/external/LibriParty/dataset/dev/`.

Generate a LibriParty-specific manifest plus repository-native annotations:

```bash
PYTHONPATH=src python scripts/generate_libriparty_manifest.py \
  --dataset-root data/external/LibriParty/dataset \
  --subset dev \
  --limit 10 \
  --output-dir outputs/libriparty_dev_manifest
```

Generated outputs:

- `outputs/libriparty_dev_manifest/manifest.csv`
- `outputs/libriparty_dev_manifest/annotations/`
- `outputs/libriparty_dev_manifest/summary.json`

The generated `manifest.csv` feeds directly into the batch runner.

Export the SpeechBrain chunk-level forward path to ONNX:

```bash
PYTHONPATH=src python scripts/export_speechbrain_onnx.py \
  --output-path outputs/onnx_export/model.onnx
```

Generated export artifacts:

- `outputs/onnx_export/model.onnx`
- `outputs/onnx_export/model.metadata.json`
- `outputs/onnx_export/model.fbank.npz` — mel filterbank matrix + Hamming window (NumPy sidecar; required by the ONNX runtime, no PyTorch needed at inference)

Profile the FP32 baseline on an existing manifest:

```bash
PYTHONPATH=src python scripts/profile_fp32_baseline.py \
  outputs/libriparty_dev_manifest/manifest.csv \
  --output-dir outputs/libriparty_dev_profile \
  --backend speechbrain_fp32
```

Profiling outputs:

- `outputs/libriparty_dev_profile/per_file.jsonl`
- `outputs/libriparty_dev_profile/summary.json`
- `outputs/libriparty_dev_profile/profile.json`

`profile.json` adds load time, wall time, RSS snapshots, CPU time, and model tensor footprint on top of the standard batch metrics.

Batch evaluation from a CSV manifest:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
  samples/batch_smoke_manifest.csv \
  --output-dir outputs/batch_run \
  --backend speechbrain_fp32
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
PYTHONPATH=src python scripts/run_batch_evaluation.py \
  samples/batch_smoke_manifest.csv \
  --output-dir outputs/batch_run \
  --backend speechbrain_fp32 \
  --save-frame-probs
```

Expected batch outputs:

- `outputs/batch_run/per_file.jsonl`
- `outputs/batch_run/summary.json`
- `outputs/batch_run/items/<id>/segments.json`
- `outputs/batch_run/items/<id>/metrics.json` when `annotation_path` is provided
- `outputs/batch_run/items/<id>/frame_probs.csv` when `--save-frame-probs` is enabled

When annotations are present, `per_file.jsonl` includes segment-overlap quality fields such as `precision`, `recall`, `f1`, `false_alarm_rate`, and `miss_rate`. `summary.json` also includes aggregate scoring totals and mean quality metrics across scored files.

Classical backend examples:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
  samples/batch_smoke_manifest.csv \
  --output-dir outputs/batch_energy_zcr \
  --backend energy_zcr
```

```bash
PYTHONPATH=src python scripts/profile_fp32_baseline.py \
  samples/batch_smoke_manifest.csv \
  --output-dir outputs/profile_webrtc_vad \
  --backend webrtc_vad
```

Dynamic INT8 backend example:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
  outputs/libriparty_dev_manifest/manifest.csv \
  --output-dir outputs/batch_dynamic_int8 \
  --backend speechbrain_dynamic_int8
```

ONNX Runtime backend example:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
  outputs/libriparty_dev_manifest/manifest.csv \
  --output-dir outputs/batch_onnx_runtime \
  --backend speechbrain_onnx_runtime \
  --onnx-model-path outputs/onnx_export/model.onnx
```

```bash
PYTHONPATH=src python scripts/profile_fp32_baseline.py \
  outputs/libriparty_dev_manifest/manifest.csv \
  --output-dir outputs/profile_onnx_runtime \
  --backend speechbrain_onnx_runtime \
  --onnx-model-path outputs/onnx_export/model.onnx
```

## Gradio demos

### Single-file ONNX demo

Upload a WAV file and visualise speech segments + frame-level probabilities (no PyTorch required):

```bash
# Prerequisites: ONNX model exported (see above)
PYTHONPATH=src python scripts/gradio_demo.py
```

Opens at http://localhost:7860.

### Multi-backend comparison demo

Compare FP32 / Dynamic INT8 / Static INT8 / ONNX side-by-side on LibriParty dev sessions,
showing RTF, F1, and first-load time:

```bash
# Prerequisites:
#   1. ONNX model exported:  PYTHONPATH=src python scripts/export_speechbrain_onnx.py --output-path outputs/onnx_export/model.onnx
#   2. LibriParty data ready: PYTHONPATH=src python scripts/bootstrap_libriparty_dataset.py --dev-sessions 10
#   3. Manifest generated:    PYTHONPATH=src python scripts/generate_libriparty_manifest.py --dataset-root data/external/LibriParty/dataset --subset dev --limit 10 --output-dir outputs/libriparty_dev_manifest
PYTHONPATH=src python scripts/gradio_compare.py
```

Opens at http://localhost:7860.  Select the number of sessions, audio duration, and which backends to include, then click **运行对比**.

Notes:

- `--backend` is available on both `scripts/run_batch_evaluation.py` and `scripts/profile_fp32_baseline.py`
- `--onnx-model-path` is required when `--backend speechbrain_onnx_runtime` is selected
- `speechbrain_fp32` produces frame probabilities
- `speechbrain_dynamic_int8` keeps the SpeechBrain model API but dynamically quantizes `GRU` and `Linear` layers at load time
- `speechbrain_onnx_runtime` uses an exported ONNX artifact, a sidecar metadata JSON, and a `.fbank.npz` sidecar; the fbank frontend is pure NumPy so PyTorch is not required at inference time
- `speechbrain_static_int8` applies post-training static quantisation; requires a calibration manifest (uses LibriParty dev set by default)
- `energy_zcr` and `webrtc_vad` currently produce speech segments only
- `requirements.txt` includes `webrtcvad-wheels`, imported in code as `webrtcvad`

The `samples/` directory is reserved for small local smoke-test audio files and is kept in the repository with a placeholder marker.
