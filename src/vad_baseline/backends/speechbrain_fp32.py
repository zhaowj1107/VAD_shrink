from vad_baseline.backends.common import BaseVADBackend
from vad_baseline.inference import (
    normalize_frame_probabilities,
    normalize_segments,
)
from vad_baseline.model import load_vad_model, model_source_name


class SpeechBrainFP32Backend(BaseVADBackend):
    backend_name = "speechbrain_fp32"
    model_name = model_source_name()
    supports_frame_probabilities = True

    def __init__(self, run_opts=None):
        self.run_opts = dict(run_opts) if run_opts else None

    def load(self):
        return load_vad_model(self.run_opts)

    def predict_segments(self, backend_model, audio_path):
        return normalize_segments(
            backend_model.get_speech_segments(str(audio_path))
        )

    def predict_frame_probabilities(self, backend_model, audio_path):
        return normalize_frame_probabilities(
            backend_model.get_speech_prob_file(str(audio_path))
        )
