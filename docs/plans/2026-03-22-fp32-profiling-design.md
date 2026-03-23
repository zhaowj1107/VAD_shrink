# FP32 Profiling Design

**Date:** 2026-03-22

## Goal

Add a reproducible FP32 profiling path that measures the current VAD baseline's resource footprint before any model shrinking or quantization work begins.

## Why This Slice

The repository can already:

- generate LibriParty manifests
- run full batch evaluation
- compute quality metrics on annotated data

What is still missing is a formal resource baseline. Without one, later `dynamic INT8` or more aggressive shrink paths cannot be evaluated rigorously.

## Scope

This first version supports:

- profiling an existing manifest-driven FP32 run
- model load timing
- batch run wall time
- process RSS and peak RSS snapshots on Linux
- CPU user and system time deltas
- model tensor footprint estimates from parameters and buffers
- writing profiling outputs alongside existing batch artifacts

Out of scope for this version:

- hardware counters
- GPU profiling
- streaming latency instrumentation
- on-disk checkpoint byte accounting from Hugging Face cache internals
- per-layer profiling

## Recommended Approach

Add a dedicated profiling module that wraps the existing batch evaluation primitives instead of rewriting the evaluation path.

Reasons:

- the repository already has working manifest parsing, model loading, per-file processing, and summary generation
- the profiling task is mainly about adding resource measurements around existing steps
- reusing the current batch path keeps quality and latency numbers aligned with the exact code path that later quantization will use

## Output Contract

The profiling run should write:

- `per_file.jsonl`
- `summary.json`
- `profile.json`

`per_file.jsonl` and `summary.json` should remain compatible with the existing batch outputs.

`profile.json` should add FP32 baseline resource information such as:

- `manifest_path`
- `num_entries`
- `load_time_sec`
- `run_wall_time_sec`
- `total_wall_time_sec`
- `rss_before_load_mb`
- `rss_after_load_mb`
- `rss_after_run_mb`
- `peak_rss_before_load_mb`
- `peak_rss_after_load_mb`
- `peak_rss_after_run_mb`
- `cpu_user_load_sec`
- `cpu_system_load_sec`
- `cpu_user_run_sec`
- `cpu_system_run_sec`
- `model_parameter_count`
- `model_parameter_bytes`
- `model_buffer_bytes`
- `model_total_tensor_bytes`
- `model_parameter_mb`
- `model_buffer_mb`
- `model_total_tensor_mb`

It should also embed or reference the batch-level quality summary so the profiling artifact is self-contained.

## Measurement Method

### Memory

Use Linux-friendly process measurements:

- current RSS from `/proc/self/status`
- peak RSS from `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`

These numbers are coarse but stable enough for a first FP32 baseline on the current machine.

### CPU Time

Use `resource.getrusage(resource.RUSAGE_SELF)` and measure deltas across:

- model loading
- batch run

### Model Footprint

Estimate model tensor footprint directly from loaded tensors:

- sum parameter `numel * element_size`
- sum buffer `numel * element_size`

This measures raw FP32 tensor footprint, which is more useful for shrink comparisons than trying to reverse-engineer cache-file layout in the current environment.

## CLI Contract

Add a new script:

- `scripts/profile_fp32_baseline.py`

Arguments:

- positional `manifest_path`
- optional `--output-dir`
- optional `--save-frame-probs`

This version deliberately profiles an existing manifest rather than combining dataset generation and evaluation into one command. The repository already has a LibriParty generator, so keeping profiling manifest-driven avoids unnecessary coupling.

## Testing Strategy

The profiling path should be covered by:

1. unit tests for model-footprint and resource-summary helpers
2. integration-style tests for the profiling runner with fake model loading and fake resource readers
3. CLI tests for parser wiring and main-path delegation

## Why This Version

This design gives the project a formal FP32 baseline for:

- memory
- CPU
- latency
- quality

That is the minimum evidence needed before any shrink or quantization claim becomes meaningful.
