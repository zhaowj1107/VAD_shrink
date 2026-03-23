# Batch Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a manifest-driven FP32 batch evaluation pipeline that runs the current offline VAD baseline over multiple WAV files and writes per-file plus aggregate outputs.

**Architecture:** Add a small batch orchestration module under `src/vad_baseline/` and a thin CLI under `scripts/`. The batch module will validate manifest input, load the model once, process files serially, write per-item artifacts, and emit JSONL plus summary outputs.

**Tech Stack:** Python 3, standard-library CSV/JSON/path handling, existing SpeechBrain/PyTorch baseline, pytest

---

### Task 1: Add manifest parsing and validation

**Files:**
- Create: `src/vad_baseline/batch.py`
- Create: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- reading a valid CSV manifest
- rejecting missing `id` / `audio_path` columns
- rejecting duplicate `id` values

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: FAIL because the batch module does not exist yet

**Step 3: Write minimal implementation**

Implement helpers that:

- read CSV manifests
- validate required columns
- validate unique ids

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: PASS

### Task 2: Add per-file batch result building

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- successful per-item processing
- failed per-item processing that records `status="failed"` and `error`

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: FAIL because the per-item processing helpers do not exist yet

**Step 3: Write minimal implementation**

Implement helpers that:

- validate the WAV before model inference
- run the existing VAD helpers
- write per-item artifacts
- return a JSON-serializable per-file result row

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: PASS

### Task 3: Add batch summary generation

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- summary counts
- mean latency / mean rtf
- p50 / p95 latency
- empty-success handling

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: FAIL because summary aggregation is incomplete

**Step 3: Write minimal implementation**

Implement a summary helper that aggregates successful per-file rows and records total/success/failure counts.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_batch.py -v`
Expected: PASS

### Task 4: Add the batch CLI entrypoint

**Files:**
- Create: `scripts/run_batch_evaluation.py`
- Create: `tests/test_batch_cli.py`
- Modify: `src/vad_baseline/io_utils.py`

**Step 1: Write the failing test**

Add tests for:

- parser wiring
- CLI main path calling batch evaluation
- JSONL output writing

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_batch_cli.py -v`
Expected: FAIL because the script does not exist yet

**Step 3: Write minimal implementation**

Implement:

- thin CLI with `manifest_path`, `--output-dir`, and optional `--save-frame-probs`
- JSONL writer helper if needed
- main path that delegates to the batch module

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_batch_cli.py -v`
Expected: PASS

### Task 5: Add smoke assets and verify end-to-end

**Files:**
- Create or modify: `samples/batch_smoke_manifest.csv`
- Modify: `README.md`

**Step 1: Write the failing test**

Add tests or checks for:

- README mentioning the batch CLI
- sample batch manifest existing

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_readme.py tests/test_batch_cli.py -v`
Expected: FAIL until docs and smoke assets exist

**Step 3: Write minimal implementation**

Add:

- a tiny sample manifest
- README batch usage example

**Step 4: Run verification**

Run:

- `PYTHONPATH=src python -m pytest tests -v`
- `PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/batch_smoke`

Expected:

- tests pass
- `per_file.jsonl` exists
- `summary.json` exists
- item artifact directories exist

### Task 6: Record follow-up gaps

**Files:**
- Modify: `DEV_LOG.MD`

**Step 1: Update development log**

Record:

- batch evaluation now exists
- current limitations
- next likely task after batch baseline

**Step 2: Verify**

Run: `sed -n '1,260p' DEV_LOG.MD`
Expected: updated next-step plan reflects the new repository state
