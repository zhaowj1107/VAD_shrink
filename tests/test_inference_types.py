from vad_baseline.inference import normalize_segments


def test_normalize_segments_computes_duration():
    raw = [(0.0, 1.25), (2.0, 3.5)]
    normalized = normalize_segments(raw)
    assert normalized == [
        {"start": 0.0, "end": 1.25, "duration": 1.25},
        {"start": 2.0, "end": 3.5, "duration": 1.5},
    ]


def test_normalize_frame_probabilities_flattens_values():
    from vad_baseline.inference import normalize_frame_probabilities

    normalized = normalize_frame_probabilities([[0.1], [0.9]])
    assert normalized == [
        {"frame_index": 0, "speech_probability": 0.1},
        {"frame_index": 1, "speech_probability": 0.9},
    ]


def test_normalize_frame_probabilities_flattens_batch_and_channel_dims():
    from vad_baseline.inference import normalize_frame_probabilities

    normalized = normalize_frame_probabilities([[[0.1], [0.9]]])
    assert normalized == [
        {"frame_index": 0, "speech_probability": 0.1},
        {"frame_index": 1, "speech_probability": 0.9},
    ]
