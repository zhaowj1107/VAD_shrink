import argparse

from vad_baseline.libriparty import generate_libriparty_manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a LibriParty manifest and repository-native annotations.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--subset",
        choices=["train", "dev", "eval", "all"],
        default="dev",
    )
    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_libriparty_manifest(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        subset=args.subset,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
