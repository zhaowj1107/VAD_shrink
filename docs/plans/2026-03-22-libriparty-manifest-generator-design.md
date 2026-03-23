# LibriParty Manifest Generator Design

**Date:** 2026-03-22

## Goal

Add a LibriParty-specific dataset ingestion tool that converts the real LibriParty directory layout into a manifest and per-item annotation files directly consumable by the repository's existing FP32 batch evaluation pipeline.

## Why This Slice

The repository can now:

- run FP32 inference on one file
- run batch evaluation from a manifest
- score annotated items with segment-overlap metrics

The main missing capability is dataset-scale input preparation. Without a generator, the current pipeline still depends on manually authored manifests and annotations, which blocks repeatable LibriParty evaluation and later FP32 vs INT8 comparisons.

## Scope

This first version supports:

- the real SpeechBrain LibriParty dataset layout already downloaded into `data/external/LibriParty`
- subset selection for `train`, `dev`, `eval`, or `all`
- optional output limiting for smoke-size subsets
- manifest generation in the repository's current CSV format
- per-session annotation conversion into the repository's current JSON segment format
- summary reporting for generated, skipped, and failed sessions

Out of scope for this version:

- generic multi-dataset support
- MUSAN integration
- frame-label generation
- threshold sweeps or `EER`
- training recipe compatibility layers
- dataset downloading logic

## Recommended Approach

Implement a LibriParty-specific generator now, but keep its internal structure ready for a future dataset-adapter layer.

Reasons:

- the directory structure and session metadata format are already known
- this is the shortest path to a research-usable evaluation baseline
- a generic dataset abstraction would add complexity before a second dataset actually exists in the repository

Alternatives considered:

1. Generic directory scanner
   - Faster to sketch, but too weak for real LibriParty metadata semantics.
2. Fully generic dataset adapter framework
   - Cleaner long term, but over-engineered for the repository's current size and needs.

## Real Dataset Structure

The downloaded archive expands into:

- `LibriParty/dataset/train/session_*`
- `LibriParty/dataset/dev/session_*`
- `LibriParty/dataset/eval/session_*`

Each session directory contains:

- `session_<n>_mixture.wav`
- `session_<n>.json`

The real session JSON format uses speaker IDs as top-level keys. Each speaker maps to a list of utterance objects with at least:

- `start`
- `stop`
- `words`
- `file`

Sessions may contain one or more speakers. The first generator version only cares about speech activity timing, not speaker identity.

## Output Contract

The generator will create an output directory such as:

- `generated/libriparty_dev/manifest.csv`
- `generated/libriparty_dev/annotations/dev_session_0.json`
- `generated/libriparty_dev/summary.json`

### Manifest

The generated CSV will follow the repository's current batch contract:

```csv
id,audio_path,annotation_path
dev_session_0,/abs/path/.../session_0_mixture.wav,/abs/path/.../annotations/dev_session_0.json
```

`id` is formed as:

- `<subset>_session_<n>`

This avoids collisions when subsets are combined under `--subset all`.

### Annotation JSON

Each generated annotation file will use the repository's current minimal speech-segment format:

```json
[
  {"start": 0.32, "end": 3.755},
  {"start": 6.534, "end": 8.979}
]
```

Generation rules:

- collect all `start/stop` pairs from all speaker utterances
- drop invalid intervals
- sort by `start`
- merge overlapping or touching intervals
- write the merged result only

Speaker identity is deliberately discarded in this version because the downstream VAD evaluation is binary speech-vs-nonspeech.

## CLI Contract

Add a new script:

- `scripts/generate_libriparty_manifest.py`

Supported arguments:

- positional or required `--dataset-root`
- required `--output-dir`
- optional `--subset` with `train`, `dev`, `eval`, or `all`
- optional `--limit`
- optional `--overwrite`

Behavior:

- `--subset all` merges all three subsets into one manifest
- `--limit` applies after subset filtering and deterministic session ordering
- existing output directories are rejected unless `--overwrite` is passed

## Internal Architecture

Add a new module:

- `src/vad_baseline/libriparty.py`

This module should expose small helpers for:

- listing session directories for a subset
- validating expected session files
- reading one LibriParty session metadata file
- extracting merged speech segments
- generating one manifest row plus one annotation payload
- writing all output files plus generation summary

The CLI remains thin and delegates all real logic to this module.

## Error Handling

Session-level failures should be recorded, not fatal:

- missing `session_<n>_mixture.wav`
- missing `session_<n>.json`
- malformed JSON
- invalid or missing `start/stop` values

Failed sessions should be skipped and counted in `summary.json`.

Fatal errors are limited to generator-level contract violations:

- dataset root missing
- requested subset folder missing
- output directory already exists without `--overwrite`

## Testing Strategy

The generator should be covered in three layers:

1. Unit tests for metadata conversion
   - one speaker
   - multiple speakers
   - overlapping utterances
   - invalid intervals

2. Integration tests for directory scanning and output generation
   - build a tiny fake LibriParty layout in `tmp_path`
   - verify `manifest.csv`, `annotations/*.json`, and `summary.json`

3. CLI tests
   - parser wiring
   - subset filtering
   - limit behavior
   - overwrite protection

## Why This Version

This design gives the repository a reliable bridge from real LibriParty data to the already-working batch evaluation baseline.

It also preserves a clean upgrade path:

- today: LibriParty-specific manifest and annotation generation
- next: dataset-native evaluation subsets at scale
- later: frame-aware metrics, `EER`, and FP32 vs INT8 comparisons on the same generated inputs
