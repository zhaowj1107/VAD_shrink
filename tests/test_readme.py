from pathlib import Path


def test_readme_mentions_run_inference_and_batch_smoke_assets_exist():
    readme = Path("README.md").read_text()
    assert "scripts/run_inference.py" in readme
    assert "scripts/run_batch_evaluation.py" in readme
    assert "scripts/generate_libriparty_manifest.py" in readme
    assert "manifest.csv" in readme
    assert "annotations/" in readme
    assert "annotation_path" in readme
    assert "metrics.json" in readme
    assert Path("samples/.gitkeep").exists()
    assert Path("samples/batch_smoke_manifest.csv").exists()
