#!/usr/bin/env python3
"""Generate soft labels from FP32 Teacher model for distillation training."""

import argparse
from pathlib import Path

from vad_baseline.distillation.config import DistillationConfig
from vad_baseline.distillation.soft_label_generator import SoftLabelGenerator
from vad_baseline.model import load_vad_model


def main():
    parser = argparse.ArgumentParser(description="Generate soft labels from Teacher model")
    parser.add_argument(
        "--train-sessions-dir",
        type=str,
        default=DistillationConfig.train_sessions_dir,
        help="Path to LibriParty train sessions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DistillationConfig.train_soft_labels_dir,
        help="Output directory for soft labels",
    )
    args = parser.parse_args()

    print("Loading Teacher model (FP32)...")
    teacher_model = load_vad_model()
    print("Teacher model loaded.")

    print(f"Scanning train sessions in {args.train_sessions_dir}...")
    train_dir = Path(args.train_sessions_dir)
    session_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(session_dirs)} train sessions")

    generator = SoftLabelGenerator(
        teacher_model=teacher_model,
        output_dir=args.output_dir,
    )

    output_paths = generator.generate_for_sessions(
        [str(d) for d in session_dirs]
    )

    print(f"\nGenerated {len(output_paths)} soft label files")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()