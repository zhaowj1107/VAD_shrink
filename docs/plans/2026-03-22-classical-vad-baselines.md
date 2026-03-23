# Classical VAD Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add backend-selectable classical VAD baselines so the repository can compare Energy+ZCR, WebRTC VAD, and SpeechBrain FP32 on the same LibriParty manifests, scoring pipeline, and profiling workflow.

**Architecture:** Introduce a narrow backend abstraction under `src/vad_baseline/backends/`, migrate the existing SpeechBrain FP32 path onto it, and update batch/profiling entrypoints to accept a backend selection. Then implement `energy_zcr` and `webrtc_vad` backends with repository-native segment outputs.

**Tech Stack:** Python 3, PyTorch/SpeechBrain for the neural backend, `webrtcvad` for the WebRTC baseline, standard-library audio orchestration, pytest

---

### Task 1: Add failing tests for the backend abstraction and SpeechBrain migration

**Files:**
- Create: `tests/test_backends.py`
- Create: `src/vad_baseline/backends/__init__.py`
- Create: `src/vad_baseline/backends/common.py`
- Create: `src/vad_baseline/backends/speechbrain_fp32.py`

**Step 1: Write the failing test**

Add tests for:

- resolving a backend by name
- SpeechBrain backend exposing a stable backend name
- SpeechBrain backend returning normalized segments through the new interface

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: FAIL because the backend abstraction does not exist yet

**Step 3: Write minimal implementation**

Implement:

- backend registry
- a thin SpeechBrain backend wrapper

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: PASS

### Task 2: Refactor batch and profiling to accept a backend

**Files:**
- Modify: `src/vad_baseline/batch.py`
- Modify: `src/vad_baseline/profiling.py`
- Modify: `tests/test_batch.py`
- Modify: `tests/test_profiling.py`

**Step 1: Write the failing test**

Add tests for:

- batch processing using a backend object instead of direct SpeechBrain helpers
- profiling preserving backend-specific model footprint info
- batch and profiling summaries recording backend name

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py tests/test_profiling.py -v`
Expected: FAIL because batch and profiling are still wired to SpeechBrain-specific helpers

**Step 3: Write minimal implementation**

Refactor the orchestration code to accept a backend instance and delegate:

- model loading / initialization
- single-file segment inference
- optional frame probabilities
- optional tensor-footprint reporting

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch.py tests/test_profiling.py -v`
Expected: PASS

### Task 3: Add the Energy+ZCR backend

**Files:**
- Create: `src/vad_baseline/backends/energy_zcr.py`
- Modify: `tests/test_backends.py`

**Step 1: Write the failing test**

Add tests for:

- detecting obvious voiced regions in a synthetic waveform
- merging nearby voiced frames into one segment
- reporting zero model tensor footprint

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: FAIL because the Energy+ZCR backend does not exist yet

**Step 3: Write minimal implementation**

Implement:

- mono/16k preprocessing
- fixed-frame energy and ZCR computation
- deterministic thresholding
- basic hangover smoothing

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: PASS

### Task 4: Add the WebRTC backend

**Files:**
- Create: `src/vad_baseline/backends/webrtc_vad.py`
- Modify: `requirements.txt`
- Modify: `tests/test_backends.py`

**Step 1: Write the failing test**

Add tests for:

- backend registration
- basic segment generation from frame decisions
- zero model tensor footprint

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: FAIL because the WebRTC backend does not exist yet

**Step 3: Write minimal implementation**

Implement:

- PCM16 conversion
- fixed WebRTC VAD settings
- frame-decision to segment conversion

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_backends.py -v`
Expected: PASS

### Task 5: Add backend-selection CLI support and documentation

**Files:**
- Modify: `scripts/run_batch_evaluation.py`
- Modify: `scripts/profile_fp32_baseline.py`
- Modify: `tests/test_batch_cli.py`
- Modify: `tests/test_profiling_cli.py`
- Modify: `README.md`
- Modify: `tests/test_readme.py`
- Modify: `DEV_LOG.MD`

**Step 1: Write the failing test**

Add checks for:

- `--backend` CLI argument
- backend selection flowing into batch/profiling runner
- README mentioning `energy_zcr` and `webrtc_vad`

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_batch_cli.py tests/test_profiling_cli.py tests/test_readme.py -v`
Expected: FAIL until the new backend-selection contract is wired and documented

**Step 3: Write minimal implementation**

Add:

- `--backend speechbrain_fp32|energy_zcr|webrtc_vad`
- doc examples
- dev log updates

**Step 4: Run verification**

Run:

- `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
- one real SpeechBrain backend run on the current `dev` manifest
- one small smoke run for `energy_zcr`
- one small smoke run for `webrtc_vad`

Expected:

- tests pass
- all three backends can produce comparable batch artifacts
