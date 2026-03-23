import inspect
import wave
from types import SimpleNamespace
from typing import Any, Mapping

MODEL_SOURCE = "speechbrain/vad-crdnn-libriparty"


def model_source_name() -> str:
    return MODEL_SOURCE


def get_audio_metadata(audio_file):
    try:
        from torchcodec.decoders import AudioDecoder

        metadata = AudioDecoder(str(audio_file)).metadata
        sample_rate = getattr(metadata, "sample_rate", None)
        duration_seconds = getattr(metadata, "duration_seconds", None)
        if sample_rate and duration_seconds is not None:
            return SimpleNamespace(
                sample_rate=int(sample_rate),
                num_frames=int(round(float(duration_seconds) * float(sample_rate))),
            )
    except Exception:
        pass

    with wave.open(str(audio_file), "rb") as handle:
        return SimpleNamespace(
            sample_rate=handle.getframerate(),
            num_frames=handle.getnframes(),
        )


def _ensure_torchaudio_backend_compat() -> None:
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: []

    if not hasattr(torchaudio, "info"):
        def compat_info(audio_file):
            return get_audio_metadata(audio_file)

        torchaudio.info = compat_info


def _ensure_huggingface_hub_compat() -> None:
    import huggingface_hub

    if "use_auth_token" in inspect.signature(
        huggingface_hub.hf_hub_download
    ).parameters:
        return

    if getattr(huggingface_hub.hf_hub_download, "_vad_shimmed", False):
        return

    original_download = huggingface_hub.hf_hub_download
    errors_module = getattr(huggingface_hub, "errors", None)
    missing_entry_error = getattr(
        errors_module,
        "RemoteEntryNotFoundError",
        None,
    )

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        try:
            return original_download(*args, **kwargs)
        except Exception as error:
            if missing_entry_error is None or not isinstance(
                error,
                missing_entry_error,
            ):
                raise

            filename = kwargs.get("filename")
            if filename is None and len(args) >= 2:
                filename = args[1]

            if filename == "custom.py":
                raise ValueError(str(error)) from error
            raise

    compat_hf_hub_download._vad_shimmed = True
    huggingface_hub.hf_hub_download = compat_hf_hub_download


def load_vad_model(
    run_opts: Mapping[str, Any] | None = None,
):
    _ensure_torchaudio_backend_compat()
    _ensure_huggingface_hub_compat()
    from speechbrain.inference.VAD import VAD

    opts = dict(run_opts) if run_opts else {}
    return VAD.from_hparams(source=MODEL_SOURCE, run_opts=opts)
