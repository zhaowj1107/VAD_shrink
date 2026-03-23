# ONNX Runtime VAD Design

**Date:** 2026-03-22

## Goal

Add an exportable `speechbrain_onnx_runtime` backend so the repository can compare the current SpeechBrain FP32 VAD baseline against a lighter ONNX Runtime inference path on the same LibriParty manifests, scoring pipeline, and profiling workflow.

## Why This Slice

The repository already has:

- a working `speechbrain_fp32` backend
- a working `speechbrain_dynamic_int8` backend
- manifest-driven batch evaluation with scoring
- backend-selectable profiling
- real LibriParty `dev` outputs for FP32, dynamic INT8, and classical baselines

The current measurements show the main bottleneck is not raw latency. The main question now is whether the repository can reduce runtime memory pressure while keeping the current neural-model quality advantage. ONNX Runtime is the right next slice because it targets inference-only deployment and can remove the need to run the SpeechBrain forward path inside the main evaluation loop.

## Scope

This version supports:

- a new export script that writes a chunk-level SpeechBrain VAD ONNX artifact plus sidecar metadata
- a new backend named `speechbrain_onnx_runtime`
- batch evaluation through `scripts/run_batch_evaluation.py`
- profiling through `scripts/profile_fp32_baseline.py`
- explicit artifact selection through a CLI argument instead of hidden export work during profiling
- CPU execution only

Out of scope for this version:

- single-file inference CLI changes
- frame-level probability export
- quantized ONNX export
- mobile-specific runtimes such as ORT Mobile or ExecuTorch
- automatic fallback export paths in the public CLI

## Current Model Reality

On this machine, the current SpeechBrain VAD object exposes:

- frontend feature extraction through `compute_features`
- input normalization through `mean_var_norm`
- neural layers through `cnn`, `rnn`, and `dnn`
- long-audio chunking and segment post-processing in Python

The key boundary is `get_speech_prob_chunk(wavs, wav_lens=None)`. That method performs:

1. feature extraction
2. normalization
3. CNN forward
4. GRU forward
5. DNN forward
6. sigmoid output

Higher-level methods such as `get_speech_prob_file()` and `get_speech_segments()` are orchestration and post-processing around that chunk-level forward. That makes chunk-level export the smallest useful ONNX boundary for this repo.

## Approach Options

### 1. Recommended: chunk-level ONNX export plus a pure-repo ORT runner

Export the chunk forward path once, then reimplement file-level chunking and segment post-processing inside the repo using ONNX Runtime and lightweight Python/Numpy logic.

Pros:

- keeps the runtime path independent from SpeechBrain model execution
- gives the cleanest memory comparison against FP32 and dynamic INT8
- keeps export cost separate from batch/profiling measurements

Cons:

- requires porting the necessary chunking and segment logic into repo code

### 2. Hybrid runtime wrapper around a loaded SpeechBrain VAD object

Load a normal SpeechBrain VAD object, replace only `get_speech_prob_chunk()` with an ORT session, and keep the rest of SpeechBrain's methods alive at runtime.

Pros:

- less new code
- lowest short-term implementation risk

Cons:

- keeps much of the SpeechBrain/PyTorch runtime alive
- weakens the main memory-shrink objective

### 3. Full pipeline rewrite/export

Push audio loading, chunking, neural forward, and post-processing all the way into a fully standalone runtime stack.

Pros:

- potentially the smallest final runtime

Cons:

- too large for the current repo phase
- much higher verification risk

## Recommended Approach

Implement option 1.

The repository already has the evaluation harness. What it lacks is a runtime path that does not depend on SpeechBrain for the actual forward pass. A pure-repo ORT runner keeps the first ONNX slice aligned with the current research question: lower memory without losing too much accuracy.

## Component Design

Create:

- `scripts/export_speechbrain_onnx.py`
- `src/vad_baseline/onnx_export.py`
- `src/vad_baseline/onnx_runtime.py`
- `src/vad_baseline/backends/speechbrain_onnx_runtime.py`

Modify:

- `src/vad_baseline/backends/__init__.py`
- `scripts/run_batch_evaluation.py`
- `scripts/profile_fp32_baseline.py`
- `README.md`
- `requirements.txt`

### Export Path

The export script loads the current FP32 SpeechBrain VAD model through the existing compatibility-aware loader, wraps the chunk forward path in a small `torch.nn.Module`, and exports that wrapper to ONNX.

The script writes:

- `model.onnx`
- `model.metadata.json`

The metadata file carries the runtime values the ORT backend needs without importing SpeechBrain at inference time, including:

- `sample_rate`
- `time_resolution`
- `source_model_name`
- export-time opset and input/output names

### Runtime Path

The ORT backend loads:

- the ONNX artifact through `onnxruntime.InferenceSession`
- the metadata sidecar

The backend then performs:

1. WAV loading from the manifest audio path
2. long-chunk splitting equivalent to the current SpeechBrain defaults
3. small-chunk batching within each large chunk
4. ORT inference on chunk batches
5. probability folding back to the full recording timeline
6. thresholding and segment post-processing equivalent to the current SpeechBrain defaults

The backend returns normalized segment rows in the same format already used by the batch evaluator.

### CLI Wiring

Both batch and profiling scripts gain:

- `--onnx-model-path`

Rules:

- optional for all non-ONNX backends
- required when `--backend speechbrain_onnx_runtime` is selected
- passed explicitly into `get_backend(...)`

This keeps export separate from runtime measurement.

## Profiling Design

The ONNX backend will not expose PyTorch tensor summaries because it does not load a PyTorch module tree for inference. Instead, backend profiling should report:

- zero PyTorch tensor footprint fields
- ONNX artifact size fields such as `model_artifact_bytes` and `model_artifact_mb`

This preserves the existing profile schema while making the backend-specific artifact size visible.

## Error Handling

Expected errors should be explicit and early:

- missing `--onnx-model-path` for the ONNX backend
- missing `.metadata.json` sidecar
- unsupported audio format for the initial WAV-only runtime path
- ORT session load failure

The initial runtime path should be explicit about the accepted input format instead of silently falling back to a heavier loader.

## Risks

### 1. Export incompatibility

The full chunk-level wrapper may hit unsupported ONNX export behavior in the installed Torch stack. If that happens, the next reduction boundary is to export only the neural core after feature extraction. That fallback is a contingency, not part of the first public interface.

### 2. Numerical drift

Even with matching logic, ONNX Runtime outputs may differ slightly from FP32 PyTorch outputs. The verification target is behavioral closeness on LibriParty `dev`, not bitwise equality.

### 3. Audio-loader scope

A WAV-only runtime path is acceptable for LibriParty and current smoke data, but it is narrower than a general-purpose audio stack.

## Testing Strategy

Testing should stay split between deterministic unit coverage and real runs.

Unit tests:

1. backend registration and loading
2. export helper wiring to `torch.onnx.export`
3. ONNX runtime backend manifest prediction with mocked ORT session outputs
4. CLI validation for `--onnx-model-path`
5. profiling summary fields for artifact-backed models

Real verification:

- full test suite
- real ONNX export on the current machine
- smoke batch and profiling run on LibriParty `dev`
- full `dev` batch and profiling comparison if smoke results are clean

## Success Criteria

This slice is successful if the repository can:

- export the current SpeechBrain VAD chunk forward path to ONNX
- run manifest-driven batch evaluation with `speechbrain_onnx_runtime`
- profile that backend with the existing profiling script
- compare accuracy, wall time, RSS, and artifact size against `speechbrain_fp32` and `speechbrain_dynamic_int8`
