# ONNX Runtime VAD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a backend-selectable ONNX Runtime VAD path that exports the SpeechBrain chunk forward once and reuses it for batch evaluation and profiling without SpeechBrain forward execution at runtime.

**Architecture:** Export a chunk-level wrapper around the current SpeechBrain `get_speech_prob_chunk()` pipeline into a standalone ONNX artifact plus metadata sidecar. Add a `speechbrain_onnx_runtime` backend that loads the artifact with ONNX Runtime, performs WAV chunking and segment post-processing inside repo code, and plugs into the existing batch and profiling flows through an explicit `--onnx-model-path` CLI argument.

**Tech Stack:** Python 3, PyTorch ONNX export, ONNX Runtime CPU, Numpy, SpeechBrain, pytest

---

### Task 1: Add failing tests for ONNX backend registration and CLI plumbing

**Files:**
- Modify: `tests/test_backends.py`
- Modify: `tests/test_batch_cli.py`
- Modify: `tests/test_profiling_cli.py`
- Modify: `src/vad_baseline/backends/__init__.py`
- Modify: `scripts/run_batch_evaluation.py`
- Modify: `scripts/profile_fp32_baseline.py`

**Step 1: Write the failing test**

Add tests for:

- backend registration under `speechbrain_onnx_runtime`
- `--onnx-model-path` parser support in batch and profiling CLIs
- CLI main functions passing `onnx_model_path` into `get_backend(...)`
- clear parser-level or main-level error when the ONNX backend is selected without `--onnx-model-path`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py tests/test_batch_cli.py tests/test_profiling_cli.py -v`
Expected: FAIL because the backend and CLI argument do not exist yet

**Step 3: Write minimal implementation**

Implement:

- backend registration in `src/vad_baseline/backends/__init__.py`
- `--onnx-model-path` in both CLI scripts
- forwarding of `onnx_model_path` into `get_backend(...)`
- clear validation for the ONNX backend path requirement

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py tests/test_batch_cli.py tests/test_profiling_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_backends.py tests/test_batch_cli.py tests/test_profiling_cli.py src/vad_baseline/backends/__init__.py scripts/run_batch_evaluation.py scripts/profile_fp32_baseline.py
git commit -m "feat: wire ONNX backend CLI inputs"
```

### Task 2: Add failing tests for export helper and implement ONNX export

**Files:**
- Create: `src/vad_baseline/onnx_export.py`
- Create: `scripts/export_speechbrain_onnx.py`
- Modify: `tests/test_model_api.py`
- Create: `tests/test_onnx_export.py`
- Modify: `requirements.txt`

**Step 1: Write the failing test**

Add tests for:

- a wrapper module that mirrors the SpeechBrain chunk forward contract
- export helper calling `torch.onnx.export(...)` with dynamic batch/time axes
- metadata sidecar writing beside the `.onnx` artifact
- export script CLI argument parsing and main call wiring

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_onnx_export.py tests/test_model_api.py -v`
Expected: FAIL because the export helper and script do not exist yet

**Step 3: Write minimal implementation**

Implement:

- a chunk-forward export wrapper in `src/vad_baseline/onnx_export.py`
- an export function that writes `model.onnx` and `model.metadata.json`
- export script CLI in `scripts/export_speechbrain_onnx.py`
- dependency entries for `onnx` and `onnxruntime`

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_onnx_export.py tests/test_model_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vad_baseline/onnx_export.py scripts/export_speechbrain_onnx.py tests/test_onnx_export.py tests/test_model_api.py requirements.txt
git commit -m "feat: add SpeechBrain ONNX export path"
```

### Task 3: Add failing tests for ORT runtime inference and implement the backend

**Files:**
- Create: `src/vad_baseline/onnx_runtime.py`
- Create: `src/vad_baseline/backends/speechbrain_onnx_runtime.py`
- Modify: `tests/test_backends.py`
- Create: `tests/test_onnx_runtime.py`
- Modify: `tests/test_batch.py`
- Modify: `tests/test_profiling.py`

**Step 1: Write the failing test**

Add tests for:

- backend load creating an ORT-backed runtime object from `onnx_model_path`
- clear failure if the model or metadata sidecar is missing
- segment prediction from mocked session probabilities
- profiling summary including zero tensor footprint plus ONNX artifact size fields
- batch/profiling execution using the new backend object

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py tests/test_onnx_runtime.py tests/test_batch.py tests/test_profiling.py -v`
Expected: FAIL because the ORT runtime implementation does not exist yet

**Step 3: Write minimal implementation**

Implement:

- ORT session loader and metadata reader in `src/vad_baseline/onnx_runtime.py`
- WAV loading, chunk batching, probability folding, thresholding, boundary extraction, close-merge, short-segment removal, and double-check logic needed for segment output
- `SpeechBrainONNXRuntimeBackend` with `supports_frame_probabilities = False`
- backend-specific size summary including `model_artifact_bytes` and `model_artifact_mb`

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py tests/test_onnx_runtime.py tests/test_batch.py tests/test_profiling.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/vad_baseline/onnx_runtime.py src/vad_baseline/backends/speechbrain_onnx_runtime.py tests/test_backends.py tests/test_onnx_runtime.py tests/test_batch.py tests/test_profiling.py
git commit -m "feat: add ONNX Runtime VAD backend"
```

### Task 4: Add docs coverage and usage examples

**Files:**
- Modify: `README.md`
- Modify: `DEV_LOG.MD`
- Modify: `tests/test_readme.py`

**Step 1: Write the failing test**

Add checks for:

- README mention of `speechbrain_onnx_runtime`
- README export command example
- README batch/profiling examples that include `--onnx-model-path`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_readme.py -v`
Expected: FAIL until the new backend and export flow are documented

**Step 3: Write minimal implementation**

Update:

- backend list in README
- export instructions
- batch/profiling command examples
- DEV log next-step and verification notes

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_readme.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add README.md DEV_LOG.MD tests/test_readme.py
git commit -m "docs: add ONNX Runtime VAD workflow"
```

### Task 5: Run full verification and real smoke/dev checks

**Files:**
- Modify: `DEV_LOG.MD`

**Step 1: Install missing runtime dependencies**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && python -m pip install onnx onnxruntime`
Expected: both packages install into the shared environment

**Step 2: Run the full test suite**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
Expected: PASS

**Step 3: Export the ONNX artifact**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/export_speechbrain_onnx.py --output-path outputs/onnx_export/model.onnx`
Expected: `outputs/onnx_export/model.onnx` and `outputs/onnx_export/model.metadata.json`

**Step 4: Run smoke batch and profiling**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx`

Expected:

- successful outputs
- profile records ONNX artifact size fields
- behavior is directionally close to FP32 smoke results

**Step 5: Run full `dev` batch and profiling if smoke is clean**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx`
- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx`

Expected:

- complete `dev` outputs
- accuracy remains close to `speechbrain_fp32`
- runtime memory and artifact size are directly comparable against FP32 and dynamic INT8

**Step 6: Record measured results**

Update `DEV_LOG.MD` with:

- export success or failure notes
- smoke results
- full `dev` batch and profiling summaries
- interpretation of memory and accuracy trade-offs
