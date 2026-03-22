from pathlib import Path


def test_project_files_exist():
    assert Path("README.md").exists()
    assert Path("requirements.txt").exists()
    assert Path("src/vad_baseline/__init__.py").exists()
