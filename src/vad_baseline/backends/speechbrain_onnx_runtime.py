from pathlib import Path

from vad_baseline.backends.common import BaseVADBackend
from vad_baseline.onnx_runtime import load_onnx_vad_runtime


def _stable_mb(value):
    return round(float(value), 6)


class SpeechBrainONNXRuntimeBackend(BaseVADBackend):
    backend_name = "speechbrain_onnx_runtime"
    model_name = "speechbrain/vad-crdnn-libriparty-onnx-runtime"
    supports_frame_probabilities = False

    def __init__(self, onnx_model_path):
        self.onnx_model_path = str(onnx_model_path)

    def load(self):
        return load_onnx_vad_runtime(self.onnx_model_path)

    def predict_segments(self, backend_model, audio_path):
        return backend_model.predict_segments(audio_path)

    def summarize_model_tensors(self, backend_model):
        return {
            "model_parameter_count": 0,
            "model_parameter_bytes": 0,
            "model_buffer_bytes": 0,
            "model_total_tensor_bytes": 0,
            "model_parameter_mb": 0.0,
            "model_buffer_mb": 0.0,
            "model_total_tensor_mb": 0.0,
            "model_artifact_bytes": int(Path(self.onnx_model_path).stat().st_size),
            "model_artifact_mb": _stable_mb(
                Path(self.onnx_model_path).stat().st_size / (1024 * 1024)
            ),
        }
