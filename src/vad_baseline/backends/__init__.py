from vad_baseline.backends.energy_zcr import EnergyZCRBackend
from vad_baseline.backends.speechbrain_dynamic_int8 import (
    SpeechBrainDynamicINT8Backend,
)
from vad_baseline.backends.speechbrain_fp32 import SpeechBrainFP32Backend
from vad_baseline.backends.speechbrain_onnx_runtime import (
    SpeechBrainONNXRuntimeBackend,
)
from vad_baseline.backends.speechbrain_static_int8 import (
    SpeechBrainStaticINT8Backend,
)
from vad_baseline.backends.webrtc_vad import WebRTCVADBackend


BACKEND_FACTORIES = {
    EnergyZCRBackend.backend_name: EnergyZCRBackend,
    SpeechBrainDynamicINT8Backend.backend_name: SpeechBrainDynamicINT8Backend,
    SpeechBrainFP32Backend.backend_name: SpeechBrainFP32Backend,
    SpeechBrainONNXRuntimeBackend.backend_name: SpeechBrainONNXRuntimeBackend,
    SpeechBrainStaticINT8Backend.backend_name: SpeechBrainStaticINT8Backend,
    WebRTCVADBackend.backend_name: WebRTCVADBackend,
}


def list_backend_names():
    return sorted(BACKEND_FACTORIES)


def get_backend(backend_name="speechbrain_fp32", **kwargs):
    try:
        backend_factory = BACKEND_FACTORIES[backend_name]
    except KeyError as error:
        raise ValueError(
            f"unknown backend: {backend_name}. available: {list_backend_names()}"
        ) from error

    return backend_factory(**kwargs)
