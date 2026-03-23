from typing import Any


def _stable_mb(value: float) -> float:
    return round(float(value), 6)


def summarize_module_tensors(modules: Any) -> dict[str, int | float]:
    if modules is None:
        parameter_count = 0
        parameter_bytes = 0
        buffer_bytes = 0
    else:
        parameters = list(modules.parameters())
        buffers = list(modules.buffers())
        parameter_count = sum(parameter.numel() for parameter in parameters)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        )
        buffer_bytes = sum(
            buffer.numel() * buffer.element_size()
            for buffer in buffers
        )

    total_tensor_bytes = parameter_bytes + buffer_bytes
    return {
        "model_parameter_count": int(parameter_count),
        "model_parameter_bytes": int(parameter_bytes),
        "model_buffer_bytes": int(buffer_bytes),
        "model_total_tensor_bytes": int(total_tensor_bytes),
        "model_parameter_mb": _stable_mb(parameter_bytes / (1024 * 1024)),
        "model_buffer_mb": _stable_mb(buffer_bytes / (1024 * 1024)),
        "model_total_tensor_mb": _stable_mb(total_tensor_bytes / (1024 * 1024)),
    }


class BaseVADBackend:
    backend_name = ""
    model_name = ""
    supports_frame_probabilities = False

    def load(self):
        raise NotImplementedError

    def predict_segments(self, backend_model, audio_path):
        raise NotImplementedError

    def predict_frame_probabilities(self, backend_model, audio_path):
        raise NotImplementedError(
            f"{self.backend_name} does not expose frame probabilities"
        )

    def summarize_model_tensors(self, backend_model):
        modules = getattr(backend_model, "mods", None)
        return summarize_module_tensors(modules)
