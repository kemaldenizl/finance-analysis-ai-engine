from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def configure_test_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage_root = tmp_path / "storage"
    input_storage = storage_root / "inputs"
    input_storage.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(storage_root))
    monkeypatch.setenv("LOCAL_INPUT_STORAGE_DIR", str(input_storage))
    monkeypatch.setenv("CLASSIFICATION_MODEL_VERSION", "rules-v1-test")

    yield
