# Dev Log

**Date:** 2026-03-22

## Current State

The repository now contains a minimal offline VAD baseline that runs `speechbrain/vad-crdnn-libriparty` on a single WAV file and writes:

- `segments.json`
- `benchmark.json`
- `frame_probs.csv` when `--save-frame-probs` is enabled

Main entrypoint:

- `scripts/run_inference.py`

Core modules:

- `src/vad_baseline/model.py`
- `src/vad_baseline/inference.py`
- `src/vad_baseline/io_utils.py`
- `src/vad_baseline/benchmark.py`

Documentation and planning:

- `README.md`
- `docs/plans/2026-03-21-vad-baseline-design.md`
- `docs/plans/2026-03-21-vad-baseline.md`

Sample assets:

- `samples/flite_hello.wav`

## What Was Implemented

1. Repository skeleton and package layout
2. Model-loading helper for `speechbrain/vad-crdnn-libriparty`
3. Output normalization helpers for speech segments and frame probabilities
4. JSON and CSV artifact writers
5. Benchmark summary helper with explicit `rtf` handling
6. CLI for offline inference on one WAV file
7. README usage instructions and sample directory placeholder
8. End-to-end smoke run on a generated speech sample

## Environment Notes

The project is currently using this shared environment instead of a local `.venv`:

- `~/Documents/Project/deepmind-research/byol/byol_env`

Additional package installed into that environment during this work:

- `torchcodec`

This was required because the installed `torchaudio 2.10.0+cu126` uses TorchCodec-backed loading.

## Compatibility Workarounds Added

The current environment has version mismatches with older SpeechBrain assumptions. Narrow compatibility shims were added in `src/vad_baseline/model.py` for:

1. `torchaudio.list_audio_backends()`
2. `torchaudio.info()`
3. `huggingface_hub.hf_hub_download(use_auth_token=...)`
4. missing default `custom.py` fetch on the HF Hub

These shims are local to the project code path and were added only to get the baseline running in the existing environment.

## Verification Evidence

Fresh unit-test run completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python -m pytest tests -v
```

Result:

- `17 passed`

Fresh end-to-end smoke run completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_inference.py samples/flite_hello.wav --output-dir outputs/flite_run --save-frame-probs
```

Generated artifacts:

- `outputs/flite_run/segments.json`
- `outputs/flite_run/benchmark.json`
- `outputs/flite_run/frame_probs.csv`

Observed smoke-run output summary:

- detected one speech segment spanning roughly the whole sample
- `audio_duration_sec`: `2.05`
- `inference_time_sec`: about `0.893`
- `rtf`: about `0.436`

## Known Limitations

1. The working stack currently depends on compatibility shims instead of a clean dependency set.
2. The baseline only supports offline single-file inference.
3. Evaluation is still smoke-test level; there is no LibriParty/MUSAN benchmark pipeline yet.
4. Quantization work has not started yet.
5. The repository is still not initialized as a git repository.

## Next Development Plan

### Phase 1: Stabilize The Baseline

1. Replace runtime compatibility shims with a confirmed compatible dependency matrix where possible.
2. Record the exact working environment in project docs, including `torchcodec` and the currently used Python environment.
3. Add one integration test around `scripts/run_inference.py` that runs on a tiny local WAV fixture and checks real artifact creation.

### Phase 2: Build Evaluation Infrastructure

1. Add manifest-generation utilities for LibriParty evaluation subsets.
2. Define a reusable evaluation output schema for per-file metrics and aggregate results.
3. Add batch evaluation scripts to run the FP32 baseline over a dataset directory or manifest.
4. Compute first research metrics such as latency summaries and VAD quality metrics on clean data.

### Phase 3: Add Noise Robustness Evaluation

1. Add MUSAN-based noise mixing utilities with configurable SNR levels.
2. Produce noisy evaluation subsets from clean speech inputs.
3. Compare baseline robustness between clean and noisy conditions.

### Phase 4: Start Quantization Work

1. Establish a fixed FP32 benchmark protocol for size, latency, and output quality.
2. Add dynamic INT8 experiments first.
3. Compare FP32 vs dynamic INT8 on the same evaluation pipeline.
4. Only after that, prototype hybrid/static quantization paths.

## Recommended Immediate Next Task

The next concrete task should be:

**Build a small batch-evaluation pipeline for the FP32 baseline on a manifest of WAV files.**

Reason:

- it turns the current smoke-test repo into a usable research baseline
- it creates the measurement path needed before quantization
- it avoids jumping into INT8 work before the evaluation layer exists
