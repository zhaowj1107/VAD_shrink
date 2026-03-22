from pathlib import Path


def test_readme_mentions_run_inference_and_samples_dir_exists():
    assert "scripts/run_inference.py" in Path("README.md").read_text()
    assert Path("samples/.gitkeep").exists()
