# Dynamic INT8 VAD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a backend-selectable dynamic INT8 SpeechBrain VAD baseline for batch evaluation and profiling on the existing LibriParty workflow.

**Architecture:** Add one new backend that reuses the current SpeechBrain VAD object but quantizes its `mods` subtree at load time with PyTorch dynamic quantization over `GRU` and `Linear`. Keep batch, profiling, scoring, and manifests unchanged so the resulting outputs remain directly comparable with `speechbrain_fp32`.

**Tech Stack:** Python 3, PyTorch dynamic quantization, SpeechBrain, pytest

---

### Task 1: Add failing tests for the dynamic INT8 backend

**Files:**
- Modify: `tests/test_backends.py`
- Create: `src/vad_baseline/backends/speechbrain_dynamic_int8.py`
- Modify: `src/vad_baseline/backends/__init__.py`

**Step 1: Write the failing test**

Add tests for:

- backend registration under `speechbrain_dynamic_int8`
- backend load calling `torch.ao.quantization.quantize_dynamic`
- backend load targeting `torch.nn.GRU` and `torch.nn.Linear`
- backend load replacing `vad_model.mods` with the quantized module tree

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: FAIL because the dynamic INT8 backend does not exist yet

**Step 3: Write minimal implementation**

Implement:

- `SpeechBrainDynamicINT8Backend`
- registry entry
- dynamic quantization helper logic inside backend load

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: PASS

### Task 2: Add CLI and documentation coverage for the new backend

**Files:**
- Modify: `tests/test_batch_cli.py`
- Modify: `tests/test_profiling_cli.py`
- Modify: `tests/test_readme.py`
- Modify: `README.md`
- Modify: `DEV_LOG.MD`

**Step 1: Write the failing test**

Add checks for:

- parser choices allowing `speechbrain_dynamic_int8`
- README mentioning `speechbrain_dynamic_int8`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch_cli.py tests/test_profiling_cli.py tests/test_readme.py -v`
Expected: FAIL until the backend name is documented

**Step 3: Write minimal implementation**

Update:

- README command examples
- dev log next-step and verification notes

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch_cli.py tests/test_profiling_cli.py tests/test_readme.py -v`
Expected: PASS

### Task 3: Run full verification and real comparison runs

**Files:**
- Modify: `DEV_LOG.MD`

**Step 1: Run the full test suite**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
Expected: PASS

**Step 2: Run a small real smoke batch**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_dynamic_int8 --backend speechbrain_dynamic_int8`
Expected: successful outputs and quality near FP32 smoke behavior

**Step 3: Run full `dev` batch and profiling**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_dynamic_int8 --backend speechbrain_dynamic_int8`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_dynamic_int8 --backend speechbrain_dynamic_int8`

Expected:

- full `dev` run completes
- batch summary comparable against `speechbrain_fp32`
- profiling summary records backend-specific tensor footprint and runtime

**Step 4: Update development log with observed INT8 results**

Record:

- full `dev` batch summary
- full `dev` profiling summary
- interpretation of whether tensor footprint moved more than total RSS
