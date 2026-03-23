# FP32 Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a manifest-driven FP32 profiling workflow that measures model load cost, memory usage, CPU time, tensor footprint, and batch evaluation timing before shrink work begins.

**Architecture:** Add a dedicated `src/vad_baseline/profiling.py` module that wraps the current batch evaluation path. Keep the CLI in `scripts/profile_fp32_baseline.py` thin. Reuse the existing batch outputs and add a top-level `profile.json` artifact with resource measurements.

**Tech Stack:** Python 3, standard-library `resource`/`pathlib`/`time`, existing batch pipeline modules, pytest

---

### Task 1: Add failing tests for profiling helpers

**Files:**
- Create: `tests/test_profiling.py`
- Create: `src/vad_baseline/profiling.py`

**Step 1: Write the failing test**

Add tests for:

- tensor-footprint estimation from parameters and buffers
- Linux RSS parsing helper
- CPU delta helper

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: FAIL because the profiling module does not exist yet

**Step 3: Write minimal implementation**

Implement helpers that:

- read RSS
- read peak RSS
- read CPU usage
- summarize parameter and buffer sizes

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: PASS

### Task 2: Add failing tests for manifest profiling orchestration

**Files:**
- Modify: `tests/test_profiling.py`
- Modify: `src/vad_baseline/profiling.py`

**Step 1: Write the failing test**

Add tests for:

- loading the model once
- profiling resource deltas across load and run
- writing `per_file.jsonl`, `summary.json`, and `profile.json`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: FAIL because the profiling runner does not exist yet

**Step 3: Write minimal implementation**

Implement a runner that:

- reads an existing manifest
- measures load and run resource deltas
- reuses `process_manifest_entry` and `summarize_results`
- writes profiling artifacts

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: PASS

### Task 3: Add the profiling CLI

**Files:**
- Create: `scripts/profile_fp32_baseline.py`
- Create: `tests/test_profiling_cli.py`

**Step 1: Write the failing test**

Add tests for:

- parser wiring
- CLI main path calling the profiling runner

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling_cli.py -v`
Expected: FAIL because the profiling CLI does not exist yet

**Step 3: Write minimal implementation**

Implement a thin CLI with:

- positional `manifest_path`
- optional `--output-dir`
- optional `--save-frame-probs`

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling_cli.py -v`
Expected: PASS

### Task 4: Document and verify the profiling workflow

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Modify: `DEV_LOG.MD`

**Step 1: Write the failing test**

Add checks for:

- README mentioning `scripts/profile_fp32_baseline.py`
- README mentioning `profile.json`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_readme.py -v`
Expected: FAIL until the docs are updated

**Step 3: Write minimal implementation**

Update docs to record:

- profiling command
- profiling outputs
- why this is the FP32 baseline before shrink work

**Step 4: Run verification**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_profile`

Expected:

- test suite passes
- profiling run writes `profile.json`
- profiling run preserves the current batch artifacts plus resource metrics
