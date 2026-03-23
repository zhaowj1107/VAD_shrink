import argparse

from vad_baseline.backends import get_backend, list_backend_names
from vad_baseline.profiling import profile_batch_manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Profile a selectable VAD backend on an existing manifest.",
    )
    parser.add_argument("manifest_path")
    parser.add_argument(
        "--output-dir",
        default="outputs/profile_run",
    )
    parser.add_argument(
        "--save-frame-probs",
        action="store_true",
    )
    parser.add_argument(
        "--backend",
        default="speechbrain_fp32",
        choices=list_backend_names(),
    )
    parser.add_argument(
        "--onnx-model-path",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        args.backend == "speechbrain_onnx_runtime"
        and not args.onnx_model_path
    ):
        parser.error("--onnx-model-path is required for speechbrain_onnx_runtime")
    backend_kwargs = {}
    if args.onnx_model_path:
        backend_kwargs["onnx_model_path"] = args.onnx_model_path
    backend = get_backend(args.backend, **backend_kwargs)
    profile_batch_manifest(
        args.manifest_path,
        args.output_dir,
        save_frame_probs=args.save_frame_probs,
        backend=backend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
