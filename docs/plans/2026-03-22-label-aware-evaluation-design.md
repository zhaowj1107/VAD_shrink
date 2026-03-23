# Label-Aware Evaluation Design

**Date:** 2026-03-22

## Goal

Add the first ground-truth-aware evaluation path to the FP32 batch baseline so the repository can compare predicted VAD speech segments against reference speech annotations and report reproducible quality metrics.

## Scope

This first version supports:

- optional per-item annotation references in the batch manifest
- reference annotations expressed as speech segments with `start` and `end`
- per-file segment-overlap metrics
- aggregate metric summaries across scored files
- failure-tolerant batch execution when annotations are missing or malformed

Out of scope for this version:

- dataset-specific LibriParty parsers
- MUSAN noise mixing
- frame-threshold sweeps
- EER computation
- parallel evaluation
- quantization

## Recommended Approach

Use segment-based reference annotations and compute duration-overlap metrics on a fixed time grid.

Reasons:

- the existing FP32 baseline already emits `segments.json`, so the output contract is segment-native
- segment annotations are simple to author and inspect
- this avoids premature dependence on a dataset-specific annotation format
- the resulting metrics are sufficient to make the FP32 baseline research-useful before INT8 work starts

Alternatives considered:

1. Frame-level scoring first
   - Better for later EER work, but the time alignment contract for current `frame_probs` is not fixed yet.
2. LibriParty-specific parser first
   - Closer to the final dataset path, but too tightly coupled before local dataset integration exists.

## Input Contract

The batch manifest keeps the current required columns:

```csv
id,audio_path
utt1,samples/a.wav
utt2,samples/b.wav
```

It may now also include:

```csv
id,audio_path,annotation_path
utt1,samples/a.wav,labels/a.json
utt2,samples/b.wav,
```

Rules:

- `annotation_path` is optional
- relative annotation paths resolve relative to the manifest directory
- files without annotations still run and produce timing artifacts
- files with annotation paths are eligible for quality scoring

## Annotation Format

The first version uses a simple JSON list of speech segments:

```json
[
  {"start": 0.20, "end": 1.10},
  {"start": 1.80, "end": 2.40}
]
```

Validation rules:

- each item must contain `start` and `end`
- `start` must be greater than or equal to `0`
- `end` must be strictly greater than `start`
- segments may be adjacent or overlapping in input, but evaluation should merge overlaps before scoring

## Metrics

Per scored file the pipeline should report:

- `reference_speech_sec`
- `predicted_speech_sec`
- `tp_sec`
- `fp_sec`
- `fn_sec`
- `precision`
- `recall`
- `f1`
- `false_alarm_rate`
- `miss_rate`
- `time_resolution_sec`

Metric definitions:

- `tp_sec`: duration where reference and prediction both mark speech
- `fp_sec`: predicted speech duration outside reference speech
- `fn_sec`: reference speech duration missed by prediction
- `precision`: `tp / (tp + fp)` when denominator is positive
- `recall`: `tp / (tp + fn)` when denominator is positive
- `f1`: harmonic mean of precision and recall when both are defined
- `false_alarm_rate`: `fp / (tp + fp)` when predicted speech is present
- `miss_rate`: `fn / (tp + fn)` when reference speech is present

## Evaluation Method

Convert both reference segments and predicted segments onto a fixed 10 ms time grid and score overlap on that grid.

This is intentionally simple. It keeps the implementation deterministic, easy to test, and close enough to later frame-level evaluation without overcommitting to current SpeechBrain internals.

## Output Contract

For any scored item, the batch runner should additionally write:

- `outputs/<run>/items/<id>/metrics.json`

Per-file batch rows should additionally include:

- `annotation_path`
- `metrics_path`
- `has_annotation`
- `scored`
- metric fields when scoring succeeds

`summary.json` should additionally include:

- `num_scored`
- mean quality metrics across scored files
- aggregate totals for `tp_sec`, `fp_sec`, and `fn_sec`

Files without annotations remain valid successes but do not contribute to quality summaries.

## Failure Behavior

Manifest-level validation still fails early for malformed CSV structure.

Per-item annotation problems should not stop the batch:

- missing annotation file
- malformed JSON
- invalid segment values

These cases should produce a failed per-file row with the error recorded, matching the existing batch failure contract.

## Why This Version

This design adds the minimum missing research capability: the repository can finally say something about VAD quality instead of only speed and artifact generation.

It also preserves a clean upgrade path:

- today: segment-based overlap metrics
- next: frame-aware scoring and threshold sweeps
- later: EER on dataset-scale manifests and FP32 vs INT8 comparison
