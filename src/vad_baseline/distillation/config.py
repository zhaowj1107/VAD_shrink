# src/vad_baseline/distillation/config.py
"""Training configuration for VAD distillation."""

from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""

    # Model architecture
    cnn_channels: tuple = (32, 64)  # 2-layer CNN with these channels
    rnn_hidden_size: int = 128  # GRU hidden size
    rnn_num_layers: int = 1
    dnn_hidden_size: int = 128

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_epochs: int = 20
    early_stopping_patience: int = 5
    gradient_clip: float = 5.0

    # Loss weights
    kl_weight: float = 0.7  # Weight for KL divergence loss
    bce_weight: float = 0.3  # Weight for BCE loss
    temperature: float = 2.0  # Soft label temperature

    # Paths
    train_soft_labels_dir: str = "data/processed/train_soft_labels"
    train_sessions_dir: str = "data/external/LibriParty/dataset/train"
    dev_sessions_dir: str = "data/external/LibriParty/dataset/dev"
    output_dir: str = "outputs/distillation"
    checkpoint_dir: str = "outputs/distillation/checkpoints"
