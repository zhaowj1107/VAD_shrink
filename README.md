# VAD Baseline Research Repo

This repository builds a minimal Linux baseline for offline voice activity detection using SpeechBrain's pretrained `vad-crdnn-libriparty` model. The repository metadata targets a CPU-oriented PyTorch baseline for Linux simulation.

Example environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate a Python environment that satisfies `requirements.txt`, then run:

```bash
PYTHONPATH=src python scripts/run_inference.py path/to/input.wav --output-dir outputs/run1
```

Optional frame-level probabilities:

```bash
PYTHONPATH=src python scripts/run_inference.py path/to/input.wav --output-dir outputs/run1 --save-frame-probs
```

Expected outputs:

- `outputs/run1/segments.json`
- `outputs/run1/benchmark.json`
- `outputs/run1/frame_probs.csv` when `--save-frame-probs` is enabled

The `samples/` directory is reserved for small local smoke-test audio files and is kept in the repository with a placeholder marker.
