from scripts import generate_libriparty_manifest


def test_libriparty_cli_requires_dataset_root_and_output_dir():
    parser = generate_libriparty_manifest.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            "data/external/LibriParty/dataset",
            "--output-dir",
            "outputs/libriparty_dev_manifest",
        ]
    )

    assert args.dataset_root == "data/external/LibriParty/dataset"
    assert args.output_dir == "outputs/libriparty_dev_manifest"
    assert args.subset == "dev"
    assert args.limit is None
    assert args.overwrite is False


def test_libriparty_cli_main_calls_generator(monkeypatch, tmp_path):
    calls = {}

    def fake_generate_libriparty_manifest(
        dataset_root,
        output_dir,
        subset="dev",
        limit=None,
        overwrite=False,
    ):
        calls["dataset_root"] = str(dataset_root)
        calls["output_dir"] = str(output_dir)
        calls["subset"] = subset
        calls["limit"] = limit
        calls["overwrite"] = overwrite
        return {"num_generated": 2}

    monkeypatch.setattr(
        generate_libriparty_manifest,
        "generate_libriparty_manifest",
        fake_generate_libriparty_manifest,
    )

    exit_code = generate_libriparty_manifest.main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "generated"),
            "--subset",
            "eval",
            "--limit",
            "5",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "dataset_root": str(tmp_path / "dataset"),
        "output_dir": str(tmp_path / "generated"),
        "subset": "eval",
        "limit": 5,
        "overwrite": True,
    }
