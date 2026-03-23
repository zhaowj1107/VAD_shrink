import platform


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def build_benchmark_summary(
    model_name: str,
    audio_duration_sec: float,
    inference_time_sec: float,
):
    audio_duration = _stable_float(audio_duration_sec)
    inference_time = _stable_float(inference_time_sec)
    if audio_duration <= 0:
        raise ValueError("audio_duration_sec must be positive")

    rtf = _stable_float(inference_time / audio_duration)

    return {
        "model_name": model_name,
        "audio_duration_sec": audio_duration,
        "inference_time_sec": inference_time,
        "rtf": rtf,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
