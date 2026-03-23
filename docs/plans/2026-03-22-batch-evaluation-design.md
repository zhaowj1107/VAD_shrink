# Batch Evaluation Design

**Date:** 2026-03-22

## Goal

Add a manifest-driven batch evaluation pipeline for the existing FP32 VAD baseline so the repository can evaluate multiple WAV files in one run and produce reproducible per-file and aggregate results.

## Scope

The first version only supports:

- CSV manifest input
- Offline FP32 evaluation
- Serial processing
- Failure-tolerant batch execution
- Per-file artifact storage
- Aggregate summary statistics

Out of scope for this version:

- Ground-truth label evaluation
- Precision / Recall / F1 / EER
- Directory scanning
- Parallel inference
- Resume / checkpointing
- Quantization

## Input Contract

The batch runner consumes a CSV manifest with exactly two required columns:

```csv
id,audio_path
utt1,samples/a.wav
utt2,samples/b.wav
```

Rules:

- `id` must be unique within the manifest
- `audio_path` must point to a readable WAV file
- No label column is required yet

## Output Contract

The batch runner writes two top-level outputs:

- `per_file.jsonl`
- `summary.json`

It also writes per-item artifacts under item-specific directories:

- `outputs/<run_name>/items/<id>/segments.json`
- `outputs/<run_name>/items/<id>/frame_probs.csv` when enabled

### Per-file Output

Each JSONL row should contain at least:

- `id`
- `audio_path`
- `status`
- `audio_duration_sec`
- `inference_time_sec`
- `rtf`
- `num_segments`
- `segments_path`
- `frame_probs_path` when present
- `error` when failed

### Summary Output

The batch summary should contain at least:

- `num_total`
- `num_success`
- `num_failed`
- `mean_inference_time_sec`
- `mean_rtf`
- `p50_inference_time_sec`
- `p95_inference_time_sec`

## Execution Flow

1. Read and validate the CSV manifest
2. Reject malformed manifests before model loading
3. Load the model once
4. Process each item serially
5. Validate WAV readability before inference on each item
6. Run VAD, write item artifacts, and record per-file results
7. Continue processing even when an item fails
8. Write `per_file.jsonl`
9. Write `summary.json`

## Failure Behavior

Batch processing must be failure tolerant.

If one item fails:

- that item is recorded with `status="failed"`
- the error message is stored in the per-file result
- processing continues for the remaining items

Manifest-level validation failures still stop the whole run early because the input contract is invalid.

## Design Rationale

This design turns the repository from a single-file smoke-test baseline into a reusable research baseline. It stays deliberately narrow so the next work can focus on dataset evaluation and later quantization comparisons instead of expanding CLI surface area too early.

The manifest-only design also aligns better with later LibriParty and MUSAN integration, where reproducible experiment splits matter more than convenience scanning.
