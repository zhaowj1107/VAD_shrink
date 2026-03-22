import sys
import types

from vad_baseline.model import MODEL_SOURCE, load_vad_model, model_source_name


def test_model_source_name_is_fixed():
    assert model_source_name() == "speechbrain/vad-crdnn-libriparty"


def test_load_vad_model_uses_fixed_source_and_run_opts(monkeypatch):
    calls = {}

    class FakeVAD:
        @classmethod
        def from_hparams(cls, source, run_opts):
            calls["source"] = source
            calls["run_opts"] = run_opts
            return "fake-vad"

    fake_module = types.ModuleType("speechbrain.inference.VAD")
    fake_module.VAD = FakeVAD
    monkeypatch.setitem(sys.modules, "speechbrain.inference.VAD", fake_module)

    result = load_vad_model({"device": "cpu"})

    assert result == "fake-vad"
    assert calls == {
        "source": MODEL_SOURCE,
        "run_opts": {"device": "cpu"},
    }


def test_load_vad_model_maps_use_auth_token_to_token(monkeypatch):
    calls = {}

    def fake_hf_hub_download(*args, token=None, **kwargs):
        calls["token"] = token
        return "downloaded"

    class FakeVAD:
        @classmethod
        def from_hparams(cls, source, run_opts):
            import huggingface_hub

            huggingface_hub.hf_hub_download(
                repo_id="speechbrain/vad-crdnn-libriparty",
                filename="hyperparams.yaml",
                use_auth_token=False,
            )
            calls["source"] = source
            calls["run_opts"] = run_opts
            return "fake-vad"

    fake_module = types.ModuleType("speechbrain.inference.VAD")
    fake_module.VAD = FakeVAD

    class FakeTorchAudio:
        list_audio_backends = staticmethod(lambda: [])

    class FakeHuggingFaceHub:
        hf_hub_download = staticmethod(fake_hf_hub_download)

    monkeypatch.setitem(sys.modules, "speechbrain.inference.VAD", fake_module)
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchAudio)
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub)

    result = load_vad_model({"device": "cpu"})

    assert result == "fake-vad"
    assert calls["token"] is False
    assert calls["source"] == MODEL_SOURCE
    assert calls["run_opts"] == {"device": "cpu"}


def test_load_vad_model_converts_missing_default_custom_py(monkeypatch):
    class FakeRemoteEntryNotFoundError(Exception):
        pass

    def fake_hf_hub_download(*args, token=None, filename=None, **kwargs):
        if filename == "custom.py":
            raise FakeRemoteEntryNotFoundError("missing custom.py")
        return "downloaded"

    class FakeErrors:
        RemoteEntryNotFoundError = FakeRemoteEntryNotFoundError

    class FakeHuggingFaceHub:
        hf_hub_download = staticmethod(fake_hf_hub_download)
        errors = FakeErrors

    class FakeVAD:
        @classmethod
        def from_hparams(cls, source, run_opts):
            import huggingface_hub

            try:
                huggingface_hub.hf_hub_download(
                    repo_id=source,
                    filename="custom.py",
                    use_auth_token=False,
                )
            except ValueError:
                return "fake-vad"

            raise AssertionError("missing default custom.py should be ignored")

    fake_module = types.ModuleType("speechbrain.inference.VAD")
    fake_module.VAD = FakeVAD

    class FakeTorchAudio:
        list_audio_backends = staticmethod(lambda: [])

    monkeypatch.setitem(sys.modules, "speechbrain.inference.VAD", fake_module)
    monkeypatch.setitem(sys.modules, "torchaudio", FakeTorchAudio)
    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHuggingFaceHub)

    result = load_vad_model({"device": "cpu"})

    assert result == "fake-vad"
