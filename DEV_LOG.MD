# Dev Log

**Date:** 2026-03-22

## Current State

The repository now contains a runnable FP32 VAD baseline with:

- single-file inference
- manifest-driven batch evaluation
- optional label-aware scoring for annotated items
- a LibriParty-specific manifest and annotation generator
- a manifest-driven FP32 profiling workflow
- a backend abstraction for interchangeable VAD predictors
- two classical comparison baselines: `energy_zcr` and `webrtc_vad`
- a dynamic quantized neural backend: `speechbrain_dynamic_int8`
- an ONNX export path plus ORT backend scaffold: `speechbrain_onnx_runtime`

Main entrypoints:

- `scripts/run_inference.py`
- `scripts/run_batch_evaluation.py`
- `scripts/generate_libriparty_manifest.py`
- `scripts/profile_fp32_baseline.py`
- `scripts/export_speechbrain_onnx.py`

Core modules:

- `src/vad_baseline/model.py`
- `src/vad_baseline/inference.py`
- `src/vad_baseline/io_utils.py`
- `src/vad_baseline/benchmark.py`
- `src/vad_baseline/batch.py`
- `src/vad_baseline/metrics.py`
- `src/vad_baseline/libriparty.py`
- `src/vad_baseline/profiling.py`
- `src/vad_baseline/backends/`

Planning and design docs:

- `docs/plans/2026-03-21-vad-baseline-design.md`
- `docs/plans/2026-03-21-vad-baseline.md`
- `docs/plans/2026-03-22-batch-evaluation-design.md`
- `docs/plans/2026-03-22-batch-evaluation.md`
- `docs/plans/2026-03-22-label-aware-evaluation-design.md`
- `docs/plans/2026-03-22-label-aware-evaluation.md`
- `docs/plans/2026-03-22-libriparty-manifest-generator-design.md`
- `docs/plans/2026-03-22-libriparty-manifest-generator.md`
- `docs/plans/2026-03-22-fp32-profiling-design.md`
- `docs/plans/2026-03-22-fp32-profiling.md`

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
11. Float-WAV compatibility fix for real LibriParty audio via TorchCodec-backed metadata
12. FP32 profiling workflow for memory, CPU, load time, and tensor footprint
13. Backend abstraction for interchangeable VAD predictors
14. Classical `energy_zcr` baseline with deterministic thresholding and smoothing
15. Classical `webrtc_vad` baseline with backend-selectable batch/profiling entrypoints
16. README and planning docs for the current baseline and evaluation workflow
17. `speechbrain_dynamic_int8` backend using dynamic quantization over `GRU` and `Linear`
18. staged runtime memory attribution for the `speechbrain_fp32` profiling path
19. chunk-level SpeechBrain ONNX export helper and CLI
20. `speechbrain_onnx_runtime` backend with explicit `--onnx-model-path` wiring
21. repo-local ORT runtime path for WAV loading, chunking, thresholding, and segment post-processing

## ONNX Runtime Status

The repository now has a working ONNX export/runtime route:

- `src/vad_baseline/onnx_export.py`
- `src/vad_baseline/onnx_runtime.py`
- `src/vad_baseline/backends/speechbrain_onnx_runtime.py`
- `scripts/export_speechbrain_onnx.py`

Current scope:

- batch evaluation
- profiling
- explicit exported model path

Current architecture:

- export boundary is the neural core after feature extraction
- runtime still computes `speechbrain_fbank` features locally in Python
- ONNX Runtime executes the normalized feature-to-probability path

Current non-goals:

- single-file inference
- frame probabilities
- automatic export during profiling

Latest measured status on this Linux machine:

- quality is effectively unchanged against `speechbrain_fp32`
- full `dev` mean `F1` is about `0.94796` for ONNX vs about `0.94806` for FP32
- full-process RSS is lower than FP32, but only modestly
- full `dev` mean inference time is much worse: about `5.06 s` vs about `2.18 s`
- the exported ONNX artifact is about `0.442 MB`, which is not smaller than the FP32 tensor footprint in practice

Interpretation:

- the current hybrid frontend + ORT-core path is useful for export experiments and artifact-based profiling
- it is not yet a strong deployment answer for "shrink on modest compute devices"
- the missing piece is removing the Python/SpeechBrain frontend from runtime

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

The repository has a LibriParty-specific generator that scans:

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

## FP32 Profiling Contract

The repository now has a profiling workflow for an existing manifest:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_profile
```

Generated outputs:

- `per_file.jsonl`
- `summary.json`
- `profile.json`

`profile.json` includes:

- `load_time_sec`
- `run_wall_time_sec`
- `total_wall_time_sec`
- current RSS snapshots
- peak RSS snapshots
- CPU user/system deltas
- model parameter and buffer footprint
- embedded batch summary

## Environment Notes

The project is currently using this shared environment instead of a local `.venv`:

- `~/Documents/Project/deepmind-research/byol/byol_env`

Additional package installed into that environment during this work:

- `torchcodec`
- `webrtcvad-wheels` needed for the `webrtc_vad` backend

This was required because the installed `torchaudio 2.10.0+cu126` uses TorchCodec-backed loading.

## Compatibility Workarounds Added

The current environment has version mismatches with older SpeechBrain assumptions. Narrow compatibility shims were added in `src/vad_baseline/model.py` for:

1. `torchaudio.list_audio_backends()`
2. `torchaudio.info()`
3. `huggingface_hub.hf_hub_download(use_auth_token=...)`
4. missing default `custom.py` fetch on the HF Hub

The `torchaudio.info()` compatibility path now prefers TorchCodec metadata so the baseline can process real LibriParty float-WAV files.

## Verification Evidence

Fresh full repository test run completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python -m pytest tests -v
```

Result:

- `89 passed`

Fresh ONNX-focused test slice also completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python -m pytest tests/test_onnx_export.py tests/test_onnx_runtime.py tests/test_backends.py tests/test_batch.py tests/test_batch_cli.py tests/test_profiling.py tests/test_profiling_cli.py -v
```

Result:

- `56 passed`

Fresh real ONNX export completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/export_speechbrain_onnx.py --output-path outputs/onnx_export/model.onnx
```

Observed artifacts:

- `outputs/onnx_export/model.onnx`
- `outputs/onnx_export/model.metadata.json`

Fresh classical backend sample-manifest smoke runs completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/smoke_energy_zcr_sample --backend energy_zcr
PYTHONPATH=src python scripts/run_batch_evaluation.py samples/batch_smoke_manifest.csv --output-dir outputs/smoke_webrtc_sample --backend webrtc_vad
```

Observed sample smoke timing:

- `energy_zcr` mean inference time: about `0.021 s`
- `webrtc_vad` mean inference time: about `0.022 s`

Fresh tiny LibriParty `dev` smoke comparison completed successfully on a 2-session subset:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_energy_zcr --backend energy_zcr
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_webrtc --backend webrtc_vad
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_speechbrain --backend speechbrain_fp32
```

Observed tiny-dev comparison:

- `energy_zcr`: mean inference about `0.053 s`, mean F1 about `0.002`
- `webrtc_vad`: mean inference about `0.090 s`, mean F1 about `0.854`
- `speechbrain_fp32`: mean inference about `2.169 s`, mean F1 about `0.974`

Fresh tiny LibriParty `dev` profiling comparison also completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_energy_zcr --backend energy_zcr
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_webrtc --backend webrtc_vad
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_speechbrain --backend speechbrain_fp32
```

Observed tiny-dev profiling highlights:

- `energy_zcr`: zero tensor footprint, total wall time about `0.743 s`
- `webrtc_vad`: zero tensor footprint, total wall time about `0.911 s`
- `speechbrain_fp32`: tensor footprint about `0.420 MB`, total wall time about `7.704 s`

Fresh full `dev` classical batch runs completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_energy_zcr --backend energy_zcr
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_webrtc_vad --backend webrtc_vad
```

Observed full `dev` classical batch summaries:

- `energy_zcr`
- output: `outputs/libriparty_dev_full_batch_energy_zcr/summary.json`
- `num_total`: `50`
- `num_success`: `50`
- `num_failed`: `0`
- `mean_inference_time_sec`: about `0.0513`
- `mean_rtf`: about `0.000176`
- `mean_precision`: about `0.964`
- `mean_recall`: about `0.00270`
- `mean_f1`: about `0.00651`

- `webrtc_vad`
- output: `outputs/libriparty_dev_full_batch_webrtc_vad/summary.json`
- `num_total`: `50`
- `num_success`: `50`
- `num_failed`: `0`
- `mean_inference_time_sec`: about `0.0786`
- `mean_rtf`: about `0.000270`
- `mean_precision`: about `0.791`
- `mean_recall`: about `0.792`
- `mean_f1`: about `0.777`

Fresh full `dev` classical profiling runs completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_energy_zcr --backend energy_zcr
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_webrtc_vad --backend webrtc_vad
```

Observed full `dev` classical profiling summaries:

- `energy_zcr`
- output: `outputs/libriparty_dev_full_profile_energy_zcr/profile.json`
- `load_time_sec`: about `0.000001`
- `run_wall_time_sec`: about `4.110`
- `total_wall_time_sec`: about `4.110`
- `rss_after_load_mb`: about `449.207`
- `rss_after_run_mb`: about `696.910`
- `peak_rss_after_run_mb`: about `715.602`
- `model_parameter_count`: `0`
- `model_total_tensor_mb`: `0.0`

Fresh dynamic INT8 smoke batch completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_dynamic_int8 --backend speechbrain_dynamic_int8
```

Observed smoke summary:

- `num_total`: `2`
- `num_success`: `2`
- `num_failed`: `0`
- `mean_inference_time_sec`: about `2.669`
- `mean_precision`: about `0.968`
- `mean_recall`: about `0.985`
- `mean_f1`: about `0.976`

Fresh ONNX Runtime smoke batch completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx
```

Observed ONNX smoke batch summary:

- output: `outputs/libriparty_dev_smoke_onnx_runtime/summary.json`
- `num_total`: `2`
- `num_success`: `2`
- `num_failed`: `0`
- `mean_inference_time_sec`: about `4.986`
- `mean_precision`: about `0.962`
- `mean_recall`: about `0.986`
- `mean_f1`: about `0.974`

Fresh ONNX Runtime smoke profiling completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx
```

Observed ONNX smoke profiling summary:

- output: `outputs/libriparty_dev_smoke_profile_onnx_runtime/profile.json`
- `load_time_sec`: about `0.656`
- `run_wall_time_sec`: about `10.222`
- `total_wall_time_sec`: about `10.878`
- `rss_after_load_mb`: about `551.281`
- `rss_after_run_mb`: about `806.176`
- `peak_rss_after_run_mb`: about `855.578`
- `model_artifact_mb`: about `0.442`

Fresh full `dev` dynamic INT8 batch completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_dynamic_int8 --backend speechbrain_dynamic_int8
```

Observed full `dev` dynamic INT8 batch summary:

- output: `outputs/libriparty_dev_full_batch_dynamic_int8/summary.json`
- `num_total`: `50`
- `num_success`: `50`
- `num_failed`: `0`
- `num_scored`: `50`
- `mean_inference_time_sec`: about `2.354`
- `mean_rtf`: about `0.00808`
- `mean_precision`: about `0.941`
- `mean_recall`: about `0.965`
- `mean_f1`: about `0.953`

Fresh full `dev` dynamic INT8 profiling completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_dynamic_int8 --backend speechbrain_dynamic_int8
```

Observed full `dev` dynamic INT8 profiling summary:

- output: `outputs/libriparty_dev_full_profile_dynamic_int8/profile.json`
- `load_time_sec`: about `1.053`
- `run_wall_time_sec`: about `128.391`
- `total_wall_time_sec`: about `129.444`
- `rss_after_load_mb`: about `559.188`
- `rss_after_run_mb`: about `855.676`
- `peak_rss_after_run_mb`: about `921.375`
- `model_parameter_count`: `21632`
- `model_total_tensor_mb`: about `0.0835`

Fresh current-format full `dev` FP32 profiling completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_fp32_current --backend speechbrain_fp32
```

Observed current-format full `dev` FP32 summary:

- output: `outputs/libriparty_dev_full_profile_fp32_current/profile.json`
- `load_time_sec`: about `1.132`
- `run_wall_time_sec`: about `111.252`
- `total_wall_time_sec`: about `112.384`
- `rss_after_load_mb`: about `558.707`
- `rss_after_run_mb`: about `923.203`
- `peak_rss_after_run_mb`: about `976.016`
- `model_total_tensor_mb`: about `0.420`

Fresh full `dev` ONNX Runtime batch completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/run_batch_evaluation.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_batch_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx
```

Observed full `dev` ONNX batch summary:

- output: `outputs/libriparty_dev_full_batch_onnx_runtime/summary.json`
- `num_total`: `50`
- `num_success`: `50`
- `num_failed`: `0`
- `num_scored`: `50`
- `mean_inference_time_sec`: about `5.063`
- `mean_rtf`: about `0.01737`
- `mean_precision`: about `0.927`
- `mean_recall`: about `0.972`
- `mean_f1`: about `0.94796`

Fresh full `dev` ONNX Runtime profiling completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_full_manifest/manifest.csv --output-dir outputs/libriparty_dev_full_profile_onnx_runtime --backend speechbrain_onnx_runtime --onnx-model-path outputs/onnx_export/model.onnx
```

Observed full `dev` ONNX profiling summary:

- output: `outputs/libriparty_dev_full_profile_onnx_runtime/profile.json`
- `load_time_sec`: about `0.716`
- `run_wall_time_sec`: about `257.648`
- `total_wall_time_sec`: about `258.364`
- `rss_after_load_mb`: about `551.469`
- `rss_after_run_mb`: about `843.781`
- `peak_rss_after_run_mb`: about `880.762`
- `model_artifact_mb`: about `0.442`

Current dynamic INT8 interpretation against FP32:

- tensor footprint dropped from about `0.420 MB` to about `0.0835 MB`
- full-process RSS improved only modestly
- inference wall time did not improve on this Linux machine and was slightly worse than FP32
- quality stayed close to FP32 and slightly exceeded the earlier FP32 run on mean `F1`

Compatibility fix added during this work:

- quantized dynamic `GRU` modules do not expose `flatten_parameters()`
- SpeechBrain's RNN wrapper expects that method
- the backend now injects a no-op compatibility shim on quantized `GRU` modules after quantization

Fresh staged FP32 runtime-memory attribution profiling completed successfully:

```bash
source ~/Documents/Project/deepmind-research/byol/byol_env/bin/activate
PYTHONPATH=src python scripts/profile_fp32_baseline.py outputs/libriparty_dev_smoke_manifest/manifest.csv --output-dir outputs/libriparty_dev_smoke_profile_fp32_attribution --backend speechbrain_fp32
```

Observed staged attribution output:

- output: `outputs/libriparty_dev_smoke_profile_fp32_attribution/profile.json`
- stages now appear under `memory_stages`
- `process_start` RSS: about `449.219 MB`
- `after_backend_load` RSS: about `557.023 MB`
- `after_first_entry_metadata` RSS: about `745.309 MB`
- `after_first_entry_inference` RSS: about `831.570 MB`
- `after_first_entry_scoring` RSS: about `831.586 MB`
- `after_full_run` RSS: about `824.449 MB`

Interpretation of the staged FP32 attribution:

- model load itself adds about `108 MB`
- the first real metadata/audio access adds another about `188 MB`
- the first true inference adds another about `86 MB`
- scoring adds essentially nothing in RSS terms
- the process retains most of that allocated memory after the run finishes

This makes the current memory picture much clearer:

- total runtime RSS is dominated by framework/runtime and first-use working-set growth
- model tensors are a tiny fraction of total process memory
- future runtime-shrink work should prioritize import/load/runtime stack reduction before chasing additional weight-only compression

Current full `dev` cross-backend interpretation:

- `speechbrain_fp32` remains the fastest neural baseline on this machine among the three neural paths tested so far
- `speechbrain_dynamic_int8` is currently the best shrink result:
  - tensor footprint dropped from about `0.420 MB` to about `0.0835 MB`
  - mean `F1` stayed close to FP32 and is slightly higher on the current `dev` run
  - runtime RSS improved, though not dramatically
- `speechbrain_onnx_runtime` preserves quality almost exactly:
  - mean `F1` about `0.94796` vs FP32 about `0.94806`
  - `rss_after_run_mb` about `843.781` vs FP32 about `923.203`
  - `peak_rss_after_run_mb` about `880.762` vs FP32 about `976.016`
  - but mean inference time worsened sharply: about `5.063 s` vs FP32 about `2.182 s`
  - and the ONNX artifact size is about the same as FP32 tensors in practice

The likely reason the ONNX route underperforms is architectural:

- the runtime still imports and runs the SpeechBrain frontend for fbank extraction
- only the neural core is executed by ONNX Runtime
- this reduces some retained process RSS but does not remove the heavy Python/runtime stack or frontend compute path

- `webrtc_vad`
- output: `outputs/libriparty_dev_full_profile_webrtc_vad/profile.json`
- `load_time_sec`: about `0.000763`
- `run_wall_time_sec`: about `9.814`
- `total_wall_time_sec`: about `9.815`
- `rss_after_load_mb`: about `449.309`
- `rss_after_run_mb`: about `695.102`
- `peak_rss_after_run_mb`: about `719.043`
- `model_parameter_count`: `0`
- `model_total_tensor_mb`: `0.0`

## Known Limitations

1. The working stack still depends on compatibility shims instead of a clean dependency matrix.
2. Label-aware scoring supports only the repository's simple JSON segment format, not dataset-native frame labels.
3. LibriParty generation is dataset-specific by design and is not yet a generic adapter framework.
4. Quality scoring is segment-grid based and does not yet expose threshold sweeps or `EER`.
5. The current profiling script measures process-level memory and CPU, not per-layer or hardware-counter metrics.
6. The ONNX path is hybrid only:
   - runtime still computes `speechbrain_fbank` features locally
   - the exported artifact is not a full end-to-end audio frontend plus neural stack
7. The ONNX backend currently supports only batch evaluation and profiling, not single-file inference or frame-probability export.
8. MUSAN integration has not started yet.
9. The project still relies on the shared `byol_env` instead of a dedicated project environment.

## Next Development Plan

1. Build a lighter frontend path for the ONNX backend:
   - either export the frontend too
   - or replace SpeechBrain fbank extraction with a lighter repo-local implementation
2. Re-run smoke and full `dev` once the frontend path is lighter, because the current ONNX route is not yet a convincing deployment path.
3. Add a repo-native comparison artifact, likely CSV or Markdown, that aligns `speechbrain_fp32`, `speechbrain_dynamic_int8`, `speechbrain_onnx_runtime`, `energy_zcr`, and `webrtc_vad` on:
   - quality
   - wall time
   - RSS
   - artifact size
4. Extend scoring with threshold sweeps and `EER`.
5. Add MUSAN-based noisy evaluation so shrink decisions are tested under noise, not only on clean LibriParty annotations.

## Recommended Immediate Next Task

The next concrete task should be:

**Remove SpeechBrain from the ONNX runtime frontend path, or export the frontend too.**

Reason:

- the current ONNX route already preserves quality
- but it is about `2.3x` slower than FP32 on full `dev`
- most of the runtime stack is still outside ONNX, so this is not yet a true shrink-for-inference solution
- fixing the frontend path is the shortest route to determine whether ONNX deserves further investment in this repo
