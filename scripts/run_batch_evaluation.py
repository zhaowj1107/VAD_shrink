import argparse

from vad_baseline.batch import run_batch_evaluation


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run batch FP32 VAD evaluation from a CSV manifest.",
    )
    parser.add_argument("manifest_path")
    parser.add_argument(
        "--output-dir",
        default="outputs/batch_run",
    )
    parser.add_argument(
        "--save-frame-probs",
        action="store_true",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_batch_evaluation(
        args.manifest_path,
        args.output_dir,
        save_frame_probs=args.save_frame_probs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
