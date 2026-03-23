import pytest

from scripts import profile_fp32_baseline


def test_profiling_cli_requires_manifest_path():
    parser = profile_fp32_baseline.build_parser()
    args = parser.parse_args(["manifest.csv"])

    assert args.manifest_path == "manifest.csv"
    assert args.output_dir == "outputs/profile_run"
    assert args.save_frame_probs is False
    assert args.backend == "speechbrain_fp32"
    assert args.onnx_model_path is None
    assert "speechbrain_dynamic_int8" in parser._option_string_actions[
        "--backend"
    ].choices
    assert "speechbrain_onnx_runtime" in parser._option_string_actions[
        "--backend"
    ].choices


def test_profiling_cli_main_calls_profiling_runner(monkeypatch, tmp_path):
    calls = {}
    fake_backend = object()

    def fake_get_backend(backend_name, **kwargs):
        calls["backend_name"] = backend_name
        calls["backend_kwargs"] = kwargs
        return fake_backend

    def fake_profile_batch_manifest(
        manifest_path,
        output_dir,
        save_frame_probs=False,
        backend=None,
    ):
        calls["manifest_path"] = str(manifest_path)
        calls["output_dir"] = str(output_dir)
        calls["save_frame_probs"] = save_frame_probs
        calls["backend"] = backend
        return {"ok": True}

    monkeypatch.setattr(profile_fp32_baseline, "get_backend", fake_get_backend)
    monkeypatch.setattr(
        profile_fp32_baseline,
        "profile_batch_manifest",
        fake_profile_batch_manifest,
    )

    exit_code = profile_fp32_baseline.main(
        [
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "profile_out"),
            "--backend",
            "webrtc_vad",
            "--save-frame-probs",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "output_dir": str(tmp_path / "profile_out"),
        "save_frame_probs": True,
        "backend_name": "webrtc_vad",
        "backend_kwargs": {},
        "backend": fake_backend,
    }


def test_profiling_cli_main_passes_onnx_model_path(monkeypatch, tmp_path):
    calls = {}
    fake_backend = object()

    def fake_get_backend(backend_name, **kwargs):
        calls["backend_name"] = backend_name
        calls["backend_kwargs"] = kwargs
        return fake_backend

    def fake_profile_batch_manifest(
        manifest_path,
        output_dir,
        save_frame_probs=False,
        backend=None,
    ):
        calls["manifest_path"] = str(manifest_path)
        calls["output_dir"] = str(output_dir)
        calls["save_frame_probs"] = save_frame_probs
        calls["backend"] = backend
        return {"ok": True}

    monkeypatch.setattr(profile_fp32_baseline, "get_backend", fake_get_backend)
    monkeypatch.setattr(
        profile_fp32_baseline,
        "profile_batch_manifest",
        fake_profile_batch_manifest,
    )

    onnx_model_path = tmp_path / "model.onnx"
    exit_code = profile_fp32_baseline.main(
        [
            str(tmp_path / "manifest.csv"),
            "--output-dir",
            str(tmp_path / "profile_out"),
            "--backend",
            "speechbrain_onnx_runtime",
            "--onnx-model-path",
            str(onnx_model_path),
        ]
    )

    assert exit_code == 0
    assert calls == {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "output_dir": str(tmp_path / "profile_out"),
        "save_frame_probs": False,
        "backend_name": "speechbrain_onnx_runtime",
        "backend_kwargs": {"onnx_model_path": str(onnx_model_path)},
        "backend": fake_backend,
    }


def test_profiling_cli_main_requires_onnx_model_path_for_onnx_backend(
    tmp_path,
    capsys,
):
    with pytest.raises(SystemExit) as error:
        profile_fp32_baseline.main(
            [
                str(tmp_path / "manifest.csv"),
                "--backend",
                "speechbrain_onnx_runtime",
            ]
        )
    assert error.value.code == 2
    assert "--onnx-model-path" in capsys.readouterr().err
