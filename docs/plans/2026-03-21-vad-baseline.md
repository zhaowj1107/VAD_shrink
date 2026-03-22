# VAD Baseline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a minimal research repo that loads a pretrained SpeechBrain VAD model, runs offline inference on a `.wav` file, writes segment outputs, and records basic Linux CPU timing.

**Architecture:** The repository starts with a small Python package under `src/vad_baseline/` and one CLI entrypoint under `scripts/`. The CLI loads a pretrained SpeechBrain VAD model, runs offline inference on one audio file, converts model output into serializable artifacts, and writes a benchmark summary for later FP32 and quantized comparisons.

**Tech Stack:** Python 3, PyTorch, SpeechBrain, torchaudio, pytest, standard-library JSON/CSV tooling

---

### Task 1: Bootstrap the repository skeleton

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `src/vad_baseline/__init__.py`
- Create: `tests/test_smoke.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_project_files_exist():
    assert Path("README.md").exists()
    assert Path("requirements.txt").exists()
    assert Path("src/vad_baseline/__init__.py").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL because the repository files do not exist yet

**Step 3: Write minimal implementation**

Create:

- `README.md` with a one-paragraph project description and the shortest intended run command
- `requirements.txt` with pinned or narrowly-ranged baseline dependencies
- `src/vad_baseline/__init__.py` as an empty package marker

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 2: Add model loading utilities

**Files:**
- Create: `src/vad_baseline/model.py`
- Create: `tests/test_model_api.py`

**Step 1: Write the failing test**

```python
from vad_baseline.model import model_source_name


def test_model_source_name_is_fixed():
    assert model_source_name() == "speechbrain/vad-crdnn-libriparty"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_model_api.py -v`
Expected: FAIL with import or missing function error

**Step 3: Write minimal implementation**

Add a small `model.py` module that:

- exposes `model_source_name()`,
- exposes a `load_vad_model()` helper,
- centralizes any SpeechBrain-specific loading parameters.

Minimal shape:

```python
MODEL_SOURCE = "speechbrain/vad-crdnn-libriparty"


def model_source_name() -> str:
    return MODEL_SOURCE
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_model_api.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 3: Add offline inference result normalization

**Files:**
- Create: `src/vad_baseline/inference.py`
- Create: `tests/test_inference_types.py`

**Step 1: Write the failing test**

```python
from vad_baseline.inference import normalize_segments


def test_normalize_segments_computes_duration():
    raw = [(0.0, 1.25), (2.0, 3.5)]
    normalized = normalize_segments(raw)
    assert normalized == [
        {"start": 0.0, "end": 1.25, "duration": 1.25},
        {"start": 2.0, "end": 3.5, "duration": 1.5},
    ]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_inference_types.py -v`
Expected: FAIL because `normalize_segments` does not exist

**Step 3: Write minimal implementation**

Implement a helper that converts raw start/end tuples into JSON-safe dictionaries with computed duration.

```python
def normalize_segments(segments):
    return [
        {
            "start": float(start),
            "end": float(end),
            "duration": float(end) - float(start),
        }
        for start, end in segments
    ]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_inference_types.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 4: Add artifact writers for JSON and CSV outputs

**Files:**
- Create: `src/vad_baseline/io_utils.py`
- Create: `tests/test_io_utils.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path

from vad_baseline.io_utils import write_json


def test_write_json_creates_parent_dirs(tmp_path):
    output = tmp_path / "nested" / "segments.json"
    write_json(output, [{"start": 0.0, "end": 1.0, "duration": 1.0}])
    assert output.exists()
    assert json.loads(output.read_text()) == [{"start": 0.0, "end": 1.0, "duration": 1.0}]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_io_utils.py -v`
Expected: FAIL because `write_json` does not exist

**Step 3: Write minimal implementation**

Implement:

- `write_json(path, payload)`
- optional `write_frame_probs_csv(path, rows)` if frame probabilities are exposed in the first pass

Use parent directory creation inside the helper.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_io_utils.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 5: Add benchmark helpers

**Files:**
- Create: `src/vad_baseline/benchmark.py`
- Create: `tests/test_benchmark.py`

**Step 1: Write the failing test**

```python
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
    assert "rtf" in summary
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_benchmark.py -v`
Expected: FAIL because the helper does not exist

**Step 3: Write minimal implementation**

Implement a helper that builds a benchmark payload with:

- `model_name`
- `audio_duration_sec`
- `inference_time_sec`
- `rtf`
- optional host metadata such as `platform` and `processor`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_benchmark.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 6: Add the CLI entrypoint

**Files:**
- Create: `scripts/run_inference.py`
- Modify: `src/vad_baseline/model.py`
- Modify: `src/vad_baseline/inference.py`
- Modify: `src/vad_baseline/io_utils.py`
- Modify: `src/vad_baseline/benchmark.py`
- Create: `tests/test_cli_args.py`

**Step 1: Write the failing test**

```python
from scripts.run_inference import build_parser


def test_cli_requires_audio_path():
    parser = build_parser()
    args = parser.parse_args(["input.wav"])
    assert args.audio_path == "input.wav"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_cli_args.py -v`
Expected: FAIL because the script or parser helper does not exist

**Step 3: Write minimal implementation**

Implement a CLI that accepts:

- positional `audio_path`
- optional `--output-dir`
- optional `--save-frame-probs`

The command should:

- load the model,
- run VAD on the input audio,
- write `segments.json`,
- write `benchmark.json`,
- optionally write `frame_probs.csv`.

Keep the script thin and move logic into `src/vad_baseline/`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_cli_args.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 7: Add one smoke-path verification target

**Files:**
- Create: `samples/.gitkeep`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_readme_mentions_run_inference():
    assert "scripts/run_inference.py" in Path("README.md").read_text()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py tests/test_readme.py -v`
Expected: FAIL because the README does not document the runnable path yet

**Step 3: Write minimal implementation**

Update `README.md` with:

- environment creation example,
- dependency installation,
- one sample inference command,
- expected output files.

Create `samples/.gitkeep` so the directory exists even before adding real audio files.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_smoke.py tests/test_readme.py -v`
Expected: PASS

**Step 5: Commit**

Skip for now because the directory is not a git repository.

### Task 8: Run repository verification

**Files:**
- Verify only

**Step 1: Run unit tests**

Run: `PYTHONPATH=src pytest tests -v`
Expected: PASS

**Step 2: Run one end-to-end inference**

Run: `PYTHONPATH=src python scripts/run_inference.py path/to/sample.wav --output-dir outputs/sample_run`
Expected:

- command exits successfully
- `outputs/sample_run/segments.json` exists
- `outputs/sample_run/benchmark.json` exists

**Step 3: Inspect artifacts**

Check:

- `segments.json` contains a list of segments
- `benchmark.json` contains timing fields

**Step 4: Record follow-up gaps**

Document any issues discovered during the first real model run, especially:

- SpeechBrain API mismatches
- torchaudio backend issues
- resampling needs
- missing sample audio

**Step 5: Commit**

Skip for now because the directory is not a git repository.

## Execution Notes

- Do not add quantization code in this plan.
- Do not add full-dataset evaluation in this plan.
- Keep the first implementation offline-only.
- Prefer narrow, testable helpers over a single large script.

## Current Blocker

This directory is not a git repository, so plan steps that mention commits should remain skipped unless `git init` is performed first.
