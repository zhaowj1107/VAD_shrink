# Label-Aware Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the FP32 batch baseline so annotated items can be scored against reference speech segments and produce reproducible per-file plus aggregate quality metrics.

**Architecture:** Keep the existing batch runner as the orchestration entrypoint, extend the manifest schema with optional `annotation_path`, and add a dedicated evaluation module under `src/vad_baseline/` for annotation loading, segment merging, grid projection, and metric computation. The batch layer will write per-item `metrics.json` when annotations are present and include aggregated quality metrics in `summary.json`.

**Tech Stack:** Python 3, standard-library JSON/CSV/path handling, existing SpeechBrain/PyTorch baseline, pytest

---

### Task 1: Add failing tests for annotation parsing and metric computation

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/vad_baseline/metrics.py`

**Step 1: Write the failing test**

Add tests for:

- reading valid annotation JSON segment lists
- rejecting invalid segments
- merging overlapping segments
- computing duration-overlap metrics on a fixed grid

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_metrics.py -v`
Expected: FAIL because the metrics module does not exist yet

**Step 3: Write minimal implementation**

Implement helpers that:

- load and validate annotation JSON
- normalize and merge segments
- project segments onto a fixed time grid
- compute per-file metrics

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_metrics.py -v`
Expected: PASS

### Task 2: Extend manifest parsing for optional annotation paths

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- reading `annotation_path` when present
- resolving relative annotation paths relative to the manifest
- preserving the current manifest behavior when `annotation_path` is absent

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: FAIL because manifest parsing does not expose annotation paths yet

**Step 3: Write minimal implementation**

Update manifest parsing so:

- `annotation_path` is optional
- empty annotation values normalize to `None`
- relative annotation paths are resolved to absolute paths

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: PASS

### Task 3: Add per-item scoring to batch processing

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `src/vad_baseline/io_utils.py`
- Modify: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- successful annotated item processing writing `metrics.json`
- per-file results including `annotation_path`, `metrics_path`, `has_annotation`, and metric fields
- unannotated items staying successful without metrics

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: FAIL because the batch pipeline does not compute or write metrics yet

**Step 3: Write minimal implementation**

Update per-item processing so:

- annotated items load reference segments
- predicted segments are scored
- `items/<id>/metrics.json` is written
- per-file rows expose scoring state and metric payload

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: PASS

### Task 4: Add aggregate quality metrics to batch summaries

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `tests/test_batch.py`

**Step 1: Write the failing test**

Add tests for:

- `num_scored`
- aggregate `tp_sec`, `fp_sec`, and `fn_sec`
- mean `precision`, `recall`, `f1`, `false_alarm_rate`, and `miss_rate`
- correct handling when no files are scored

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: FAIL because the summary currently only reports timing metrics

**Step 3: Write minimal implementation**

Extend the summary builder to aggregate scored rows separately from plain successful rows and emit quality metrics only when present.

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py -v`
Expected: PASS

### Task 5: Document and verify the new scoring path

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Modify: `DEV_LOG.MD`

**Step 1: Write the failing test**

Add checks for:

- README mentioning `annotation_path`
- README mentioning `metrics.json`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_readme.py -v`
Expected: FAIL until the docs are updated

**Step 3: Write minimal implementation**

Update docs to record:

- optional annotated manifest usage
- new per-item metric artifact
- next planned step after this feature

**Step 4: Run verification**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/batch_smoke`

Expected:

- test suite passes
- batch run still succeeds without annotations
- annotated runs write `metrics.json` and extended summaries when applicable
