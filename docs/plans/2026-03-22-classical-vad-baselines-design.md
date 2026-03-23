# Classical VAD Baselines Design

**Date:** 2026-03-22

## Goal

Add two non-neural VAD baselines before dynamic INT8 work so the repository can compare classical, engineering-oriented VAD approaches against the current neural FP32 baseline and later INT8 variants.

## Why This Slice

The repository already has:

- a working SpeechBrain FP32 baseline
- LibriParty dataset ingestion
- batch evaluation and scoring
- FP32 profiling

What is missing is a meaningful non-neural reference point. Without one, later INT8 results can only be compared against the original neural model, not against lightweight traditional alternatives.

## Scope

This version supports:

- a minimal backend abstraction for interchangeable VAD predictors
- migration of the current SpeechBrain FP32 path onto that abstraction
- one classical energy-based baseline using short-time energy, zero-crossing rate, and hangover smoothing
- one WebRTC VAD baseline using the `webrtcvad` Python binding
- reuse of the existing manifest, scoring, and profiling workflows

Out of scope for this version:

- GMM-HMM or other more complex classical VAD systems
- streaming APIs
- threshold sweeps for all classical methods
- automatic hyperparameter search
- dynamic INT8 implementation

## Recommended Approach

Introduce a backend abstraction now and hang all VAD methods off it.

Reasons:

- the existing batch and profiling flows are currently wired directly to SpeechBrain
- adding multiple comparison baselines without an abstraction would create a second round of refactoring later when INT8 is added
- a single backend interface keeps scoring, profiling, and manifests fully comparable across methods

## Comparison Set

The repository should aim for this comparison ladder:

- `energy_zcr_baseline`
- `webrtc_vad_baseline`
- `speechbrain_fp32`
- `speechbrain_dynamic_int8` later

This creates a clean progression:

- extremely lightweight traditional baseline
- stronger traditional engineering baseline
- neural FP32 reference
- neural compressed reference later

## Backend Interface

Add a small backend layer under:

- `src/vad_baseline/backends/`

Expected components:

- `common.py`
- `speechbrain_fp32.py`
- `energy_zcr.py`
- `webrtc_vad.py`

The minimal interface should support:

- backend identification
- optional backend loading / initialization
- running VAD on one file and returning repository-native speech segments
- optional frame-probability support when a backend can provide it
- optional model-footprint reporting for profiling

The interface should stay deliberately narrow so the current batch and profiling code can remain mostly unchanged.

## Energy + ZCR Baseline

### Input assumptions

- convert audio to mono
- resample to 16 kHz
- use fixed frame length, recommended first version: `20 ms`

### Decision rule

Per frame compute:

- short-time energy
- zero-crossing rate

Use a simple deterministic thresholding rule:

- energy threshold relative to the file's frame-energy distribution
- ZCR as a secondary gate for rejecting some low-energy noise-like frames

### Post-processing

Apply minimal smoothing:

- merge adjacent speech frames
- minimum speech duration
- minimum silence duration
- hangover extension

The target is not maximum quality. The target is a stable, explainable, lightweight classical baseline.

## WebRTC VAD Baseline

Use the `webrtcvad` Python binding with a fixed first-version configuration:

- sample rate: `16000`
- frame size: `20 ms`
- aggressiveness: `2`

The backend should:

- convert audio to mono PCM16
- apply frame-wise WebRTC VAD
- merge speech frames into segments
- reuse the same post-processing policy shape as the energy baseline where reasonable

This gives a realistic stronger classical reference without expanding the project into a full traditional-signal-processing benchmark suite.

## Integration Points

The backend abstraction should be consumed by:

- `src/vad_baseline/batch.py`
- `src/vad_baseline/profiling.py`

The LibriParty manifest generator and annotation path should remain unchanged.

The batch and profiling CLIs should gain a backend selection flag so the same manifest can be evaluated with:

- `speechbrain_fp32`
- `energy_zcr`
- `webrtc_vad`

The single-file inference CLI can optionally follow later, but the initial comparison work only needs batch and profiling support.

## Profiling Expectations

Traditional baselines likely have:

- negligible model tensor footprint
- much lower memory overhead
- lower or comparable wall time
- weaker quality than the neural baseline

The profiling output should represent that clearly by allowing backends with:

- `0` parameter bytes
- `0` buffer bytes

## Testing Strategy

Testing should be staged:

1. backend abstraction tests
2. SpeechBrain migration tests
3. energy baseline unit tests
4. WebRTC backend unit tests
5. batch/profiling backend-selection tests

This version should prefer deterministic tests using tiny synthetic waveforms and mocked backend loading over broad end-to-end runs for every method.

## Why This Version

This design gives the repository exactly the comparison context needed before quantization:

- how small and cheap can a very simple traditional baseline be?
- how strong is a mature traditional engineering baseline?
- how far is the neural FP32 baseline from those trade-offs?
- does neural INT8 later close that gap?
