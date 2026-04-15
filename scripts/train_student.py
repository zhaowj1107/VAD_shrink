#!/usr/bin/env python3
"""Train Student model via knowledge distillation."""

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from vad_baseline.distillation.config import DistillationConfig
from vad_baseline.distillation.dataset import (
    LibriPartyDistillationDataset,
    collate_distillation_batch,
)
from vad_baseline.distillation.student_model import SimplifiedCRDNN
from vad_baseline.distillation.trainer import VADDistillationTrainer
from vad_baseline.model import load_vad_model


def main():
    parser = argparse.ArgumentParser(description="Train Student VAD via distillation")
    parser.add_argument(
        "--soft-labels-dir",
        type=str,
        default=DistillationConfig.train_soft_labels_dir,
        help="Directory containing soft labels from Teacher",
    )
    parser.add_argument(
        "--train-sessions-dir",
        type=str,
        default=DistillationConfig.train_sessions_dir,
        help="Path to LibriParty train sessions",
    )
    parser.add_argument(
        "--dev-sessions-dir",
        type=str,
        default=DistillationConfig.dev_sessions_dir,
        help="Path to LibriParty dev sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DistillationConfig.output_dir,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DistillationConfig.max_epochs,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DistillationConfig.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DistillationConfig.learning_rate,
        help="Learning rate",
    )
    args = parser.parse_args()

    config = DistillationConfig()
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.max_epochs = args.epochs

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("VAD Distillation Training")
    print("=" * 60)
    print(f"Train sessions: {args.train_sessions_dir}")
    print(f"Dev sessions: {args.dev_sessions_dir}")
    print(f"Soft labels: {args.soft_labels_dir}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print()

    # Load Teacher model
    print("Loading Teacher model (FP32)...")
    teacher_model = load_vad_model()
    print("Teacher model loaded.")

    # Create Student model
    print("Creating Student model (Simplified CRDNN)...")
    student_model = SimplifiedCRDNN(
        input_size=80,  # 80 mel bins
        cnn_channels=config.cnn_channels,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_num_layers=config.rnn_num_layers,
        dnn_hidden_size=config.dnn_hidden_size,
    )

    # Count parameters
    param_count = sum(p.numel() for p in student_model.parameters())
    print(f"Student parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # Create trainer
    trainer = VADDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        config=config,
    )

    # Load train dataset
    print("\nLoading train dataset...")
    train_dir = Path(args.train_sessions_dir)
    train_session_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(train_session_dirs)} train sessions")

    train_dataset = LibriPartyDistillationDataset(
        session_dirs=[str(d) for d in train_session_dirs],
        soft_labels_dir=args.soft_labels_dir,
    )
    print(f"Train dataset: {len(train_dataset)} sessions")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_distillation_batch,
        num_workers=0,
    )

    # Load dev dataset
    print("\nLoading dev dataset...")
    dev_dir = Path(args.dev_sessions_dir)
    dev_session_dirs = sorted([d for d in dev_dir.iterdir() if d.is_dir()])
    print(f"Found {len(dev_session_dirs)} dev sessions")

    dev_dataset = LibriPartyDistillationDataset(
        session_dirs=[str(d) for d in dev_session_dirs],
        soft_labels_dir=args.soft_labels_dir,
    )
    print(f"Dev dataset: {len(dev_dataset)} sessions")

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_distillation_batch,
        num_workers=0,
    )

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Train loss: {train_metrics['train_loss']:.4f} "
              f"(KL: {train_metrics['train_kl']:.4f}, BCE: {train_metrics['train_bce']:.4f})")

        # Evaluate
        dev_metrics = trainer.eval(dev_loader)
        print(f"Dev F1: {dev_metrics['dev_f1']:.4f} "
              f"(P: {dev_metrics['dev_precision']:.4f}, R: {dev_metrics['dev_recall']:.4f})")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch + 1, dev_metrics["dev_f1"])
        print(f"Saved checkpoint: {checkpoint_path}")

        # Early stopping check
        if dev_metrics["dev_f1"] > trainer.best_dev_f1:
            trainer.best_dev_f1 = dev_metrics["dev_f1"]
            trainer.patience_counter = 0
            best_path = checkpoint_dir / "best.pt"
            trainer.save_checkpoint(best_path, epoch + 1, dev_metrics["dev_f1"])
            print(f"New best! Saved best.pt with F1: {dev_metrics['dev_f1']:.4f}")
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print("\n" + "=" * 60)
    print(f"Training complete. Best dev F1: {trainer.best_dev_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
