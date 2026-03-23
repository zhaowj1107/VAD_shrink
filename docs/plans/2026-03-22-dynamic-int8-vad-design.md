# Dynamic INT8 VAD Design

**Date:** 2026-03-22

## Goal

Add a `speechbrain_dynamic_int8` backend so the repository can compare the current SpeechBrain FP32 VAD baseline against a low-effort dynamic quantized variant on the same LibriParty manifests, scoring pipeline, and profiling workflow.

## Why This Slice

The repository now already has:

- a working `speechbrain_fp32` backend
- backend-selectable batch evaluation
- backend-selectable profiling
- classical comparison baselines
- full `dev` LibriParty evaluation outputs for FP32 and classical methods

What is missing is the first neural compression baseline. Dynamic INT8 is the right first slice because it is the lowest-risk quantization variant and can be integrated without adding calibration or retraining.

## Scope

This version supports:

- a new backend named `speechbrain_dynamic_int8`
- dynamic quantization of the current SpeechBrain VAD model at load time
- reuse of the existing batch, scoring, manifest, and profiling flows
- direct comparison against `speechbrain_fp32`, `webrtc_vad`, and `energy_zcr`

Out of scope for this version:

- static INT8 quantization
- quantization-aware training
- saved quantized artifacts
- single-file inference CLI changes
- exporting to ONNX, TorchScript, TFLite, or other runtimes

## Current Model Reality

A quick probe of the currently loaded SpeechBrain VAD model on this machine shows the module stack includes:

- `GRU`
- `Linear`
- `Conv2d`
- feature-extraction and normalization blocks

That is important because PyTorch dynamic quantization primarily helps recurrent and linear layers, not convolution-heavy frontends. A second probe confirmed that on this machine `torch.ao.quantization.quantize_dynamic(...)` really converts the current model's `GRU` and `Linear` submodules into dynamic quantized variants with packed weights.

## Recommended Approach

Add a thin backend that starts from the existing SpeechBrain FP32 backend contract and only changes model loading.

Reasons:

- the current backend abstraction already isolates model loading from batch and profiling orchestration
- dynamic INT8 does not need calibration data or training loops
- the repository already has the exact evaluation infrastructure needed for a first quantized baseline
- this keeps the implementation small and makes later static INT8 work easier

## Backend Design

Create:

- `src/vad_baseline/backends/speechbrain_dynamic_int8.py`

Backend contract:

- `backend_name = "speechbrain_dynamic_int8"`
- `model_name = "speechbrain/vad-crdnn-libriparty-dynamic-int8"`
- preserve frame-probability support
- CPU-only assumption

Loading behavior:

1. load the existing SpeechBrain VAD model using the current compatibility-aware `load_vad_model()`
2. obtain `vad_model.mods`
3. apply dynamic quantization to:
   - `torch.nn.GRU`
   - `torch.nn.Linear`
4. assign the quantized module tree back to `vad_model.mods`
5. return the modified model object

Prediction behavior stays identical to the FP32 backend because the model API surface does not change.

## Integration Points

The new backend should plug into:

- `src/vad_baseline/backends/__init__.py`
- existing CLI backend choices through `list_backend_names()`
- existing batch and profiling entrypoints without new orchestration code

No manifest, annotation, or scoring changes are needed.

## Profiling Expectations

Expected improvements:

- reduced tensor footprint for quantized `GRU` and `Linear` weights
- possibly lower CPU wall time on the current Linux machine

Expected non-improvements:

- total RSS may remain dominated by Python, PyTorch, SpeechBrain, feature extraction, and general runtime overhead
- convolutional layers remain unquantized in this first version

This means the dynamic INT8 result should be interpreted mainly as:

- a neural compression baseline
- a precision-retention check
- a lower-effort quantization reference before static INT8

not as the final edge-deployment answer.

## Risks

### 1. Limited memory win

The repository's prior FP32 profiling showed a tiny model tensor footprint compared with large process RSS. Dynamic INT8 may shrink the tensor footprint while barely moving total process memory.

### 2. Partial quantization only

Because the current model contains convolutional and frontend signal-processing components, only part of the network is quantized in this first pass.

### 3. Upstream PyTorch API churn

`torch.ao.quantization.quantize_dynamic` emits a deprecation warning in the installed PyTorch version. It still works on this machine today, but the code should keep the quantization logic isolated so later migration to `torchao` is straightforward.

## Testing Strategy

Testing should stay narrow and deterministic:

1. backend registration test
2. backend load test proving dynamic quantization is invoked on `GRU` and `Linear`
3. backend load test proving the quantized modules are written back to the SpeechBrain wrapper object
4. CLI and README checks for the new backend name

Real verification should then run:

- full test suite
- a small real smoke run on LibriParty `dev`
- full `dev` batch and profiling comparison if the smoke run is clean

## Why This Version

This version gives the repository the next comparison rung with the least engineering risk:

- `energy_zcr`
- `webrtc_vad`
- `speechbrain_fp32`
- `speechbrain_dynamic_int8`

That is enough to answer the next research question:

- does low-effort neural quantization preserve the FP32 quality advantage while reducing model footprint and possibly runtime cost?
