# Dev Log

**Date:** 2026-03-22

## Current State

The repository now contains a runnable FP32 VAD baseline with:

- single-file inference
- manifest-driven batch evaluation
- optional label-aware scoring for annotated items
- a LibriParty-specific manifest and annotation generator

Main entrypoints:

- `scripts/run_inference.py`
- `scripts/run_batch_evaluation.py`
- `scripts/generate_libriparty_manifest.py`

Core modules:

- `src/vad_baseline/model.py`
- `src/vad_baseline/inference.py`
- `src/vad_baseline/io_utils.py`
- `src/vad_baseline/benchmark.py`
- `src/vad_baseline/batch.py`
- `src/vad_baseline/metrics.py`
- `src/vad_baseline/libriparty.py`

Planning and design docs:

- `docs/plans/2026-03-21-vad-baseline-design.md`
- `docs/plans/2026-03-21-vad-baseline.md`
- `docs/plans/2026-03-22-batch-evaluation-design.md`
- `docs/plans/2026-03-22-batch-evaluation.md`
- `docs/plans/2026-03-22-label-aware-evaluation-design.md`
- `docs/plans/2026-03-22-label-aware-evaluation.md`
- `docs/plans/2026-03-22-libriparty-manifest-generator-design.md`
- `docs/plans/2026-03-22-libriparty-manifest-generator.md`

Local data:

- `data/external/LibriParty.tar.gz`
- `data/external/LibriParty/dataset`

Sample assets:

- `samples/flite_hello.wav`
- `samples/batch_smoke_manifest.csv`

## What Was Implemented

1. Repository skeleton and package layout
2. SpeechBrain model-loading helper for `speechbrain/vad-crdnn-libriparty`
3. Output normalization helpers for speech segments and frame probabilities
4. JSON, JSONL, and CSV artifact writers
5. Benchmark summary helper with stable float output
6. CLI for offline inference on one WAV file
7. Manifest-driven batch evaluation pipeline
8. Optional per-item label-aware evaluation based on speech-segment annotations
9. Aggregate batch summaries for timing and quality metrics
10. LibriParty-specific manifest and annotation generator
11. README and planning docs for the current baseline and evaluation workflow

## Label-Aware Evaluation Contract

The batch manifest supports an optional `annotation_path` column:

```csv
id,audio_path,annotation_path
utt1,samples/a.wav,labels/a.json
utt2,samples/b.wav,
```

The current repository annotation format is a simple JSON list of speech segments:

```json
[
  {"start": 0.20, "end": 1.10},
  {"start": 1.80, "end": 2.40}
]
```

Annotated items produce:

- `items/<id>/metrics.json`

Per-file outputs expose:

- `annotation_path`
- `metrics_path`
- `has_annotation`
- `scored`
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

Current scoring method:

- merge reference and predicted speech segments
- project both onto a fixed `10 ms` grid
- compute overlap-based duration metrics

## LibriParty Generator Contract

The repository now has a LibriParty-specific generator that scans:

- `data/external/LibriParty/dataset/train`
- `data/external/LibriParty/dataset/dev`
- `data/external/LibriParty/dataset/eval`

Expected session layout:

- `session_<n>/session_<n>_mixture.wav`
- `session_<n>/session_<n>.json`

Generated outputs:

- `manifest.csv`
- `annotations/<id>.json`
- `summary.json`

Generated ids are subset-qualified, for example:

- `dev_session_0`
- `train_session_51`

Current LibriParty conversion method:

- read every top-level speaker list in a session JSON
- extract all `start/stop` utterance intervals
- merge overlapping or adjacent speech segments
- write repository-native binary speech annotations

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

- `43 passed`

Fresh label-aware batch smoke run completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py /tmp/vad_label_smoke/manifest.csv --output-dir outputs/batch_labeled_smoke
```

Generated artifacts:

- `outputs/batch_labeled_smoke/per_file.jsonl`
- `outputs/batch_labeled_smoke/summary.json`
- `outputs/batch_labeled_smoke/items/flite_hello/segments.json`
- `outputs/batch_labeled_smoke/items/flite_hello/metrics.json`

Observed labeled smoke summary:

- `num_total`: `1`
- `num_success`: `1`
- `num_failed`: `0`
- `num_scored`: `1`
- `mean_inference_time_sec`: about `0.712`
- `mean_rtf`: about `0.347`
- `mean_precision`: `1.0`
- `mean_recall`: about `0.990`
- `mean_f1`: about `0.995`

Fresh LibriParty generator smoke run completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/generate_libriparty_manifest.py --dataset-root data/external/LibriParty/dataset --subset dev --limit 2 --output-dir outputs/libriparty_dev_manifest
```

Generated artifacts:

- `outputs/libriparty_dev_manifest/manifest.csv`
- `outputs/libriparty_dev_manifest/annotations/dev_session_0.json`
- `outputs/libriparty_dev_manifest/annotations/dev_session_1.json`
- `outputs/libriparty_dev_manifest/summary.json`

Observed LibriParty generator summary:

- `subset`: `dev`
- `num_sessions_found`: `50`
- `num_generated`: `2`
- `num_failed`: `0`
- `num_skipped`: `48`

## Known Limitations

1. The working stack still depends on compatibility shims instead of a clean dependency matrix.
2. Label-aware scoring supports only the repository's simple JSON segment format, not dataset-native frame labels.
3. LibriParty generation is dataset-specific by design and is not yet a generic adapter framework.
4. Quality scoring is segment-grid based and does not yet expose threshold sweeps or `EER`.
5. MUSAN integration has not started yet.
6. Quantization work has not started yet.
7. The project still relies on the shared `byol_env` instead of a dedicated project environment.

## Next Development Plan

### Phase 1: Dataset Evaluation At Scale

1. Add a convenience path for generating full `train`, `dev`, `eval`, or `all` manifests into stable output folders.
2. Add one follow-up command or doc flow that runs generated manifests directly through the batch baseline.
3. Record stable experiment naming conventions for manifest-generation and evaluation outputs.

### Phase 2: Stronger Evaluation Metrics

1. Extend scoring to support frame-aware evaluation.
2. Add threshold sweeps over frame probabilities.
3. Compute research-facing metrics such as `EER`.

### Phase 3: Noise Robustness Evaluation

1. Add MUSAN-based noise mixing utilities with configurable SNR levels.
2. Produce noisy evaluation subsets from clean speech inputs.
3. Compare clean vs noisy robustness on the same scoring pipeline.

### Phase 4: Quantization Work

1. Freeze a stable FP32 benchmark protocol for size, latency, and quality.
2. Add dynamic INT8 experiments first.
3. Compare FP32 vs dynamic INT8 on the same generated and labeled evaluation pipeline.
4. Only after that, prototype hybrid/static quantization paths.

## Recommended Immediate Next Task

The next concrete task should be:

**Add a direct evaluation flow from generated LibriParty manifests into the existing batch baseline.**

Reason:

- the repository can now ingest real LibriParty sessions and emit repository-native annotations
- the shortest path to useful research output is to connect generated manifests to repeatable FP32 evaluation runs
- frame-level metrics and quantization experiments become more meaningful once end-to-end dataset evaluation is routine
