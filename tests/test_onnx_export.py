import json
from types import SimpleNamespace

import torch


class _FakeComputeFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, wavs):
        self.calls.append(tuple(wavs.shape))
        batch_size = wavs.shape[0]
        return torch.ones((batch_size, 4, 3), dtype=torch.float32)


class _FakeMeanVarNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, feats, wav_lens):
        self.calls.append(
            (tuple(feats.shape), tuple(wav_lens.shape), wav_lens.tolist())
        )
        return feats + 1.0


class _FakeCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, feats):
        self.calls.append(tuple(feats.shape))
        batch_size = feats.shape[0]
        return torch.ones((batch_size, 4, 2, 3), dtype=torch.float32)


class _FakeRNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs):
        self.calls.append(tuple(inputs.shape))
        batch_size, num_frames, _ = inputs.shape
        return torch.zeros((batch_size, num_frames, 1), dtype=torch.float32), None


class _FakeDNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs):
        self.calls.append(tuple(inputs.shape))
        return inputs + 2.0


class _FakeMods(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_features = _FakeComputeFeatures()
        self.mean_var_norm = _FakeMeanVarNorm()
        self.cnn = _FakeCNN()
        self.rnn = _FakeRNN()
        self.dnn = _FakeDNN()


def test_chunk_export_wrapper_matches_speechbrain_chunk_contract():
    from vad_baseline.onnx_export import SpeechBrainChunkExportWrapper

    vad_model = SimpleNamespace(mods=_FakeMods())
    wrapper = SpeechBrainChunkExportWrapper(vad_model)

    output = wrapper(
        torch.ones((1, 16000), dtype=torch.float32),
        torch.zeros((1,), dtype=torch.float32),
    )

    assert output.shape == (1, 4, 1)
    assert torch.allclose(output, torch.sigmoid(torch.full((1, 4, 1), 2.0)))
    assert vad_model.mods.compute_features.calls == [(1, 16000)]
    assert vad_model.mods.mean_var_norm.calls == [((1, 4, 3), (1,), [1.0])]
    assert vad_model.mods.cnn.calls == [(1, 4, 3)]
    assert vad_model.mods.rnn.calls == [(1, 4, 6)]
    assert vad_model.mods.dnn.calls == [(1, 4, 1)]


def test_export_speechbrain_onnx_calls_export_and_writes_metadata(
    tmp_path,
):
    from vad_baseline.onnx_export import export_speechbrain_onnx

    calls = {}

    class FakeVADModel:
        def __init__(self):
            self.mods = _FakeMods()
            self.sample_rate = 16000
            self.time_resolution = 0.01

    def fake_load_model_fn(run_opts=None):
        calls["run_opts"] = run_opts
        return FakeVADModel()

    def fake_export_fn(model, args, output_path, **kwargs):
        calls["model_type"] = type(model).__name__
        calls["args_shapes"] = [tuple(arg.shape) for arg in args]
        calls["output_path"] = str(output_path)
        calls["kwargs"] = kwargs
        output_path.write_bytes(b"fake-onnx")

    output_path = tmp_path / "model.onnx"
    result = export_speechbrain_onnx(
        output_path=output_path,
        run_opts={"device": "cpu"},
        load_model_fn=fake_load_model_fn,
        export_fn=fake_export_fn,
        sample_frames=32000,
        opset_version=17,
    )

    metadata_path = tmp_path / "model.metadata.json"
    assert calls["run_opts"] == {"device": "cpu"}
    assert calls["model_type"] == "SpeechBrainCoreExportWrapper"
    assert calls["args_shapes"] == [(1, 4, 3)]
    assert calls["output_path"] == str(output_path)
    assert calls["kwargs"]["input_names"] == ["feats"]
    assert calls["kwargs"]["output_names"] == ["speech_probabilities"]
    assert calls["kwargs"]["dynamic_axes"] == {
        "feats": {0: "batch_size", 1: "num_frames"},
        "speech_probabilities": {0: "batch_size", 1: "num_frames"},
    }
    assert calls["kwargs"]["opset_version"] == 17
    assert output_path.read_bytes() == b"fake-onnx"
    assert metadata_path.exists()
    assert json.loads(metadata_path.read_text()) == {
        "source_model_name": "speechbrain/vad-crdnn-libriparty",
        "sample_rate": 16000,
        "time_resolution": 0.01,
        "input_names": ["feats"],
        "output_names": ["speech_probabilities"],
        "opset_version": 17,
        "frontend": "speechbrain_fbank",
    }
    assert result == {
        "model_path": str(output_path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "source_model_name": "speechbrain/vad-crdnn-libriparty",
        "sample_rate": 16000,
        "time_resolution": 0.01,
        "input_names": ["feats"],
        "output_names": ["speech_probabilities"],
        "opset_version": 17,
        "frontend": "speechbrain_fbank",
    }


def test_export_sentence_norm_matches_full_length_sentence_normalization():
    from vad_baseline.onnx_export import normalize_sentence_norm_for_export

    class FakeSentenceNorm:
        norm_type = "sentence"
        mean_norm = True
        std_norm = True
        eps = 1e-10

    feats = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )

    normalized = normalize_sentence_norm_for_export(feats, FakeSentenceNorm())

    expected = torch.tensor(
        [
            [[-0.70710677, -0.70710677], [0.70710677, 0.70710677]],
            [[-0.70710677, -0.70710677], [0.70710677, 0.70710677]],
        ]
    )
    assert torch.allclose(normalized, expected, atol=1e-6)


def test_export_cli_main_calls_export_helper(monkeypatch, tmp_path):
    from scripts import export_speechbrain_onnx

    calls = {}

    def fake_export_speechbrain_onnx(output_path, opset_version=17):
        calls["output_path"] = str(output_path)
        calls["opset_version"] = opset_version
        return {"model_path": str(output_path)}

    monkeypatch.setattr(
        export_speechbrain_onnx,
        "export_speechbrain_onnx",
        fake_export_speechbrain_onnx,
    )

    exit_code = export_speechbrain_onnx.main(
        [
            "--output-path",
            str(tmp_path / "model.onnx"),
            "--opset-version",
            "19",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "output_path": str(tmp_path / "model.onnx"),
        "opset_version": 19,
    }
