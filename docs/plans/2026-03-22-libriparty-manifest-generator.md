# LibriParty Manifest Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a LibriParty-specific manifest generator that scans the real dataset layout, converts session metadata into repository-native annotation JSON files, and writes a batch-runner-compatible manifest.

**Architecture:** Add a dedicated `src/vad_baseline/libriparty.py` module for dataset scanning, session validation, and annotation extraction. Keep the CLI in `scripts/generate_libriparty_manifest.py` thin. Reuse the existing CSV/JSON writing helpers so the generated outputs feed directly into the current batch evaluation path.

**Tech Stack:** Python 3, standard-library JSON/CSV/pathlib/shutil handling, existing repository helpers, pytest

---

### Task 1: Add failing tests for LibriParty annotation extraction

**Files:**
- Create: `tests/test_libriparty.py`
- Create: `src/vad_baseline/libriparty.py`

**Step 1: Write the failing test**

Add tests for:

- extracting merged speech segments from a one-speaker session JSON
- extracting merged speech segments from a multi-speaker session JSON
- rejecting invalid `start/stop` intervals

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: FAIL because the LibriParty module does not exist yet

**Step 3: Write minimal implementation**

Implement helpers that:

- load one session JSON
- iterate speaker utterance lists
- extract `start/stop` pairs
- reuse merged segment normalization compatible with repository annotations

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: PASS

### Task 2: Add failing tests for subset scanning and manifest row generation

**Files:**
- Modify: `tests/test_libriparty.py`
- Modify: `src/vad_baseline/libriparty.py`

**Step 1: Write the failing test**

Add tests for:

- listing sessions for a requested subset
- generating stable ids like `dev_session_0`
- rejecting missing session WAV or session JSON

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: FAIL because directory scanning and session validation are incomplete

**Step 3: Write minimal implementation**

Implement helpers that:

- scan subset directories deterministically
- validate expected files
- build manifest rows and annotation payloads

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: PASS

### Task 3: Add failing tests for full output generation

**Files:**
- Modify: `tests/test_libriparty.py`
- Modify: `src/vad_baseline/libriparty.py`
- Modify: `src/vad_baseline/io_utils.py`

**Step 1: Write the failing test**

Add tests for:

- generating `manifest.csv`
- generating `annotations/<id>.json`
- generating `summary.json`
- honoring `--limit`
- refusing to overwrite existing output without explicit permission

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: FAIL because end-to-end generator output does not exist yet

**Step 3: Write minimal implementation**

Implement the main generation function that:

- resolves subset selection
- writes annotation files
- writes manifest CSV
- writes summary JSON
- supports `overwrite`

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty.py -v`
Expected: PASS

### Task 4: Add the LibriParty generator CLI

**Files:**
- Create: `scripts/generate_libriparty_manifest.py`
- Create: `tests/test_libriparty_cli.py`

**Step 1: Write the failing test**

Add tests for:

- parser wiring
- CLI main path calling the generator with parsed arguments

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty_cli.py -v`
Expected: FAIL because the script does not exist yet

**Step 3: Write minimal implementation**

Implement a thin CLI that accepts:

- `--dataset-root`
- `--output-dir`
- `--subset`
- `--limit`
- `--overwrite`

and delegates to the LibriParty module.

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_libriparty_cli.py -v`
Expected: PASS

### Task 5: Document and verify the real-data path

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Modify: `DEV_LOG.MD`

**Step 1: Write the failing test**

Add checks for:

- README mentioning `generate_libriparty_manifest.py`
- README mentioning the generated `manifest.csv` and `annotations/`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_readme.py -v`
Expected: FAIL until docs mention the new generator

**Step 3: Write minimal implementation**

Update docs to record:

- where LibriParty is stored locally
- how to generate manifests
- what the generator writes
- what the next research step is

**Step 4: Run verification**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/generate_libriparty_manifest.py --dataset-root data/external/LibriParty/dataset --subset dev --limit 2 --output-dir outputs/libriparty_dev_manifest`

Expected:

- test suite passes
- generator writes `manifest.csv`
- generator writes `annotations/*.json`
- generator writes `summary.json`
