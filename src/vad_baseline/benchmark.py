import platform


def build_benchmark_summary(
    model_name: str,
    audio_duration_sec: float,
    inference_time_sec: float,
):
    audio_duration = float(audio_duration_sec)
    inference_time = float(inference_time_sec)
    if audio_duration <= 0:
        raise ValueError("audio_duration_sec must be positive")

    rtf = inference_time / audio_duration

    return {
        "model_name": model_name,
        "audio_duration_sec": audio_duration,
        "inference_time_sec": inference_time,
        "rtf": rtf,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
