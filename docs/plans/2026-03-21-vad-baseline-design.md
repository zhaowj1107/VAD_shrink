# VAD Baseline Design

**Date:** 2026-03-21

## Goal

Build the smallest useful research repository that can run a pretrained neural VAD model on a Linux machine, save speech-segment outputs, and capture baseline timing data. This baseline will become the reference point for later dataset evaluation and INT8 quantization work.

## Scope

Phase 1 only covers offline file inference with a pretrained SpeechBrain VAD model in FP32.

Included:

- Reproducible Python environment setup
- Loading `speechbrain/vad-crdnn-libriparty`
- Running VAD on one input `.wav`
- Saving detected speech segments to a machine-readable file
- Optionally saving frame-level probabilities
- Recording basic runtime measurements on Linux CPU

Excluded from Phase 1:

- Full LibriParty or MUSAN evaluation
- EER benchmarking
- Dynamic or static INT8 quantization
- Streaming / real-time inference
- Model training or fine-tuning
- ARM / NEON deployment claims

## Why Start Here

The repository is currently empty, so the first priority is to establish a working baseline rather than designing the full quantization pipeline upfront. Starting from the pretrained SpeechBrain VAD keeps the project aligned with the later research target while avoiding unnecessary early complexity.

This also follows the standard VAD pipeline described in the Aalto speech processing material: compute frame-level speech evidence, apply thresholds and post-processing, and produce speech segments. The SpeechBrain model already provides this path with a modern neural backend.

## Recommended Approach

Use the pretrained `vad-crdnn-libriparty` model as the first runnable baseline.

Reasons:

- It is directly relevant to the later quantization study.
- It avoids spending time on training before inference infrastructure exists.
- It produces realistic speech-segment outputs instead of only a toy signal-processing baseline.
- It gives a clear FP32 reference for later `dynamic INT8` and `hybrid/static` comparisons.

Alternatives considered:

1. Rule-based VAD using energy thresholds
   - Simpler, but weakly connected to the planned CRDNN quantization study.
2. Training a custom lightweight VAD
   - More controllable, but far too much setup for the first milestone.

## First Milestone

The first milestone is complete when the repository can:

- install required dependencies,
- run one command against a `.wav` file,
- load the pretrained SpeechBrain VAD,
- emit speech segment timestamps,
- save results under an output directory,
- report a small benchmark summary.

## Repository Shape

The initial repository layout should be:

- `docs/plans/`
- `README.md`
- `requirements.txt`
- `samples/`
- `scripts/run_inference.py`
- `src/vad_baseline/`
- `outputs/` via runtime creation, not committed artifacts

Suggested internal Python modules:

- `src/vad_baseline/model.py`
  - model loading and caching
- `src/vad_baseline/inference.py`
  - audio-to-segments pipeline
- `src/vad_baseline/io_utils.py`
  - output serialization helpers
- `src/vad_baseline/benchmark.py`
  - timing and environment capture

## Data Strategy

No full research dataset is required yet.

Phase 1 only needs one to three `.wav` samples to validate the path:

`input wav -> pretrained VAD -> speech segments -> saved outputs`

LibriParty and MUSAN should only be introduced after baseline inference is stable and reproducible.

## Output Contract

The first version should write the following artifacts:

- `segments.json`
  - detected speech segments with `start`, `end`, and `duration`
- `frame_probs.csv`
  - optional frame-level probabilities if exposed cleanly
- `benchmark.json`
  - model name, audio duration, inference time, and host CPU metadata

This keeps outputs easy to inspect and easy to reuse in later evaluation scripts.

## Validation Criteria

Phase 1 validation is functional, not publication-grade.

Success means:

- the command runs to completion,
- output files are created consistently,
- repeated runs are stable,
- the detected speech segments roughly match audible speech in the sample file.

Failure at this stage means the infrastructure is not ready for dataset benchmarking or quantization work.

## Risks And Constraints

- SpeechBrain model download may depend on network availability.
- Audio format mismatches may require resampling or channel conversion.
- x86/Linux timing results are useful for baseline comparison but do not prove ARM edge-device performance.
- Exact output APIs from SpeechBrain may require adaptation after inspecting the installed version.

## Phase 2 Preview

Once the baseline runs reliably, the next research stage can add:

- LibriParty evaluation manifests
- MUSAN-based noisy test generation
- FP32 benchmark protocol
- `dynamic INT8` experiments
- `hybrid/static` quantization experiments
- metric comparison across accuracy, size, and latency

## Notes

This repository is not currently a git repository, so the design is recorded locally but cannot be committed yet. If version control is needed, initialize git before implementation begins.
