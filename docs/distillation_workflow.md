# VAD Distillation Workflow

## Overview

This document describes the complete workflow for training a distilled VAD model.

## Prerequisites

- LibriParty dataset at `data/external/LibriParty/dataset/`
- Python dependencies: `pip install -r requirements.txt`

## Step 1: Generate Soft Labels

Run the Teacher model (FP32) on training data to generate soft labels:

```bash
PYTHONPATH=src python scripts/generate_soft_labels.py \
    --train-sessions-dir data/external/LibriParty/dataset/train \
    --output-dir data/processed/train_soft_labels
```

This will:
- Load the FP32 Teacher model
- Process each training session
- Save frame-level probabilities as .npy files

## Step 2: Train Student Model

Train the Simplified CRDNN student model:

```bash
PYTHONPATH=src python scripts/train_student.py \
    --soft-labels-dir data/processed/train_soft_labels \
    --train-sessions-dir data/external/LibriParty/dataset/train \
    --dev-sessions-dir data/external/LibriParty/dataset/dev \
    --output-dir outputs/distillation \
    --epochs 20 \
    --batch-size 8
```

This will:
- Load Teacher and Student models
- Train with KL + BCE loss
- Evaluate on dev set each epoch
- Save checkpoints with best F1

## Step 3: Evaluate

Evaluate the trained model using the existing evaluation pipeline:

```bash
PYTHONPATH=src python scripts/run_batch_evaluation.py \
    data/libriparty_dev_manifest.csv \
    --output-dir outputs/distilled_eval \
    --backend distilled \
    --checkpoint outputs/distillation/checkpoints/best.pt
```

## Expected Results

| Metric | Target |
|--------|--------|
| F1 (dev) | >= 0.90 |
| Inference time | < 1s per session |
| Parameters | < 0.5M |
