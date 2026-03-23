# Runtime Memory Attribution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing SpeechBrain FP32 profiling workflow with staged runtime memory attribution so `profile.json` explains where process memory grows during load and the first real inference pass.

**Architecture:** Keep the current profiling entrypoint and summary fields, but add staged RSS/CPU snapshots around model load and the first manifest entry. Restrict the feature to the `speechbrain_fp32` path so the implementation stays narrow and directly addresses the current analysis need.

**Tech Stack:** Python 3, existing repository profiling helpers, pytest

---

### Task 1: Add failing tests for stage snapshots

**Files:**
- Modify: `tests/test_profiling.py`
- Modify: `src/vad_baseline/profiling.py`

**Step 1: Write the failing test**

Add tests for:

- `profile_batch_manifest()` emitting `memory_stages` for the SpeechBrain FP32 path
- stage names appearing in the expected order
- `delta_rss_from_previous_mb` and `delta_rss_from_start_mb` being calculated correctly
- `memory_stages` omitted for explicit non-SpeechBrain backends

**Step 2: Run test to verify it fails**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: FAIL because staged attribution does not exist yet

**Step 3: Write minimal implementation**

Implement:

- stage snapshot helper
- first-entry instrumentation path
- stage gating for `speechbrain_fp32`

**Step 4: Run test to verify it passes**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests/test_profiling.py -v`
Expected: PASS

### Task 2: Run full verification and real FP32 profiling

**Files:**
- Modify: `DEV_LOG.MD`

**Step 1: Run the full test suite**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python -m pytest tests -v`
Expected: PASS

**Step 2: Run a real FP32 profiling pass**

Run: `source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate && PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_fp32_attribution --backend speechbrain_fp32`
Expected: profile completes and `profile.json` contains `memory_stages`

**Step 3: Record findings**

Update `DEV_LOG.MD` with:

- the new staged output location
- the observed memory growth pattern across stages
- the main interpretation for future runtime-shrink work
