from pathlib import Path

from scripts import run_batch_evaluation


def test_batch_cli_requires_manifest_path():
    parser = run_batch_evaluation.build_parser()
    args = parser.parse_args(["manifest.csv"])
    assert args.manifest_path == "manifest.csv"


def test_batch_cli_main_calls_batch_runner(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,audio_path\nutt1,samples/flite_hello.wav\n")
    calls = {}

    def fake_run_batch_evaluation(
        manifest_path,
        output_dir,
        save_frame_probs=False,
    ):
        calls["manifest_path"] = str(manifest_path)
        calls["output_dir"] = str(output_dir)
        calls["save_frame_probs"] = save_frame_probs
        return {"num_total": 1}

    monkeypatch.setattr(
        run_batch_evaluation,
        "run_batch_evaluation",
        fake_run_batch_evaluation,
    )

    exit_code = run_batch_evaluation.main(
        [
            str(manifest),
            "--output-dir",
            str(tmp_path / "batch_out"),
            "--save-frame-probs",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "manifest_path": str(manifest),
        "output_dir": str(tmp_path / "batch_out"),
        "save_frame_probs": True,
    }
