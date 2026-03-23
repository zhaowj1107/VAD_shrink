from vad_baseline.model import get_audio_metadata


def _to_python(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def normalize_segments(segments):
    segment_rows = _to_python(segments)
    if not segment_rows:
        return []

    if not isinstance(segment_rows[0], (list, tuple)):
        segment_rows = list(zip(segment_rows[::2], segment_rows[1::2]))

    normalized = []
    for start, end in segment_rows:
        start_value = float(start)
        end_value = float(end)
        normalized.append(
            {
                "start": start_value,
                "end": end_value,
                "duration": end_value - start_value,
            }
        )
    return normalized


def _flatten_probabilities(values):
    python_values = _to_python(values)
    if isinstance(python_values, (list, tuple)):
        flattened = []
        for item in python_values:
            flattened.extend(_flatten_probabilities(item))
        return flattened

    return [python_values]


def normalize_frame_probabilities(frame_probabilities):
    flattened = _flatten_probabilities(frame_probabilities)
    normalized = []
    for frame_index, probability in enumerate(flattened):
        normalized.append(
            {
                "frame_index": frame_index,
                "speech_probability": float(probability),
            }
        )
    return normalized


def get_wav_duration_sec(audio_path):
    metadata = get_audio_metadata(audio_path)
    sample_rate = metadata.sample_rate
    frame_count = metadata.num_frames

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    return frame_count / sample_rate


def run_vad_on_file(vad_model, audio_path):
    audio_file = str(audio_path)
    return normalize_segments(vad_model.get_speech_segments(audio_file))


def get_frame_probabilities_for_file(vad_model, audio_path):
    return normalize_frame_probabilities(
        vad_model.get_speech_prob_file(str(audio_path))
    )
