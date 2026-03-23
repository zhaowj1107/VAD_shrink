import argparse

from vad_baseline.onnx_export import (
    DEFAULT_ONNX_OPSET,
    export_speechbrain_onnx,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export the SpeechBrain VAD chunk forward path to ONNX.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/onnx_export/model.onnx",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_ONNX_OPSET,
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    export_speechbrain_onnx(
        output_path=args.output_path,
        opset_version=args.opset_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
