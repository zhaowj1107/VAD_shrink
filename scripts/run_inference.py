import argparse
from pathlib import Path
from time import perf_counter

from vad_baseline.benchmark import build_benchmark_summary
from vad_baseline.inference import (
    get_frame_probabilities_for_file,
    get_wav_duration_sec,
    run_vad_on_file,
)
from vad_baseline.io_utils import write_frame_probs_csv, write_json
from vad_baseline.model import load_vad_model, model_source_name


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run offline SpeechBrain VAD on one wav file.",
    )
    parser.add_argument("audio_path")
    parser.add_argument(
        "--output-dir",
        default="outputs/latest",
    )
    parser.add_argument(
        "--save-frame-probs",
        action="store_true",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    audio_path = Path(args.audio_path)
    output_dir = Path(args.output_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    audio_duration_sec = get_wav_duration_sec(audio_path)

    vad_model = load_vad_model()
    started_at = perf_counter()
    segments = run_vad_on_file(vad_model, audio_path)
    inference_time_sec = perf_counter() - started_at

    write_json(output_dir / "segments.json", segments)
    if args.save_frame_probs:
        frame_probabilities = get_frame_probabilities_for_file(
            vad_model,
            audio_path,
        )
        write_frame_probs_csv(
            output_dir / "frame_probs.csv",
            frame_probabilities,
        )

    benchmark = build_benchmark_summary(
        model_name=model_source_name(),
        audio_duration_sec=audio_duration_sec,
        inference_time_sec=inference_time_sec,
    )
    write_json(output_dir / "benchmark.json", benchmark)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
