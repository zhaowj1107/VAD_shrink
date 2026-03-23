# Runtime Memory Attribution Design

**Date:** 2026-03-22

## Goal

Add staged runtime memory attribution to the existing profiling workflow so the repository can explain where the `speechbrain_fp32` process memory grows during profiling on real LibriParty manifests.

## Why This Slice

The repository already has end-to-end profiling numbers for:

- total load time
- total run wall time
- RSS snapshots before load, after load, and after run
- total tensor footprint

Those numbers are enough to prove that model weights are not the dominant memory cost, but they are not enough to explain where the runtime memory actually grows. The next useful step is not a full-blown external memory profiler. The next useful step is staged attribution inside the current workflow.

## Scope

This version supports:

- staged RSS, peak RSS, and CPU snapshots inside `profile_batch_manifest()`
- only the `speechbrain_fp32` path
- reuse of the existing `profile.json` output
- deterministic unit tests using fake readers and fake manifest processing

Out of scope:

- support for classical backends
- support for `speechbrain_dynamic_int8`
- per-layer or per-op memory attribution
- external profiling tools such as `memray` or `torch.profiler`
- changes to single-file inference

## Recommended Approach

Extend the current profiling flow with a small stage-recording helper and record memory/CPU snapshots at a few meaningful lifecycle points.

Reasons:

- minimal change surface
- keeps all profiling results in the existing `profile.json`
- gives actionable answers without invasive hooks into SpeechBrain internals
- avoids overbuilding a custom memory instrumentation framework too early

## Stages

The first version should record these stages:

- `process_start`
- `after_backend_load`
- `after_first_entry_metadata`
- `after_first_entry_inference`
- `after_first_entry_scoring`
- `after_full_run`

These stages answer the key questions:

- how expensive is model loading itself?
- how much memory appears only after the first real audio metadata access?
- how much additional memory appears after the first true network inference?
- does scoring and artifact generation add meaningful memory?
- how much memory remains resident after the full batch completes?

## Output Design

Add a new field to `profile.json`:

- `memory_stages`

Each item should contain:

- `stage_name`
- `rss_mb`
- `peak_rss_mb`
- `cpu_user_sec`
- `cpu_system_sec`
- `delta_rss_from_previous_mb`
- `delta_rss_from_start_mb`

The top-level summary fields stay unchanged so existing consumers remain compatible.

## Implementation Strategy

Use a small local helper in `profiling.py` to capture a stage snapshot using the already injected readers:

- `current_rss_reader`
- `peak_rss_reader`
- `cpu_times_reader`

Do not change batch orchestration broadly. Instead:

1. capture `process_start` before model load
2. capture `after_backend_load` after `load_model_fn()`
3. process the first manifest entry manually with a wrapped timer so the code can capture:
   - after metadata
   - after inference
   - after scoring
4. process the remaining entries through the existing path
5. capture `after_full_run`

This approach keeps the stage boundaries explicit while avoiding invasive hooks throughout the codebase.

## Risk

### 1. First-entry instrumentation only approximates steady-state

The first real item may include one-time allocations that later files reuse. That is acceptable because those one-time costs are exactly what this profiler is intended to expose.

### 2. Slight duplication between profiling and batch logic

Instrumenting the first entry may require a small amount of local orchestration duplication. That is acceptable if kept narrow and test-covered.

### 3. Existing output compatibility

The implementation must only add fields, not rename or remove existing profile fields.

## Testing Strategy

Unit tests should cover:

- stage ordering
- stage values from fake readers
- correct delta calculations
- gating so `memory_stages` appear only for `speechbrain_fp32`

Real verification should re-run the existing FP32 profiling command on a real manifest and inspect the new `memory_stages` output.

## Why This Version

This version gives the repository a practical answer to the current research question:

- if total process RSS is huge but model tensors are tiny, where does the runtime memory actually accumulate?

It does so with minimal engineering cost and without muddying the study with external profiler complexity.
