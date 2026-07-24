import json
import os
import warnings
from pathlib import Path

import pytest

from pptagent.model_utils import ModelManager
from pptagent.utils import Config


# warning of zipfile indicates that presentation save failed
def pytest_configure() -> None:
    warnings.filterwarnings("error", module=r"zipfile")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip credentialed legacy tests unless they are explicitly enabled."""
    if os.environ.get("PPTAGENT_RUN_LLM_TESTS") == "1":
        return
    marker = pytest.mark.skip(
        reason="set PPTAGENT_RUN_LLM_TESTS=1 to run credentialed model tests"
    )
    for item in items:
        if item.get_closest_marker("llm") is not None:
            item.add_marker(marker)


class TestConfig:
    """Shared legacy fixtures with lazy model initialization."""

    def __init__(self) -> None:
        package_dir = Path(__file__).resolve().parents[1]
        self.template = str(package_dir / "templates" / "default")
        self.document = str(package_dir / "test" / "fixtures" / "document")
        self.ppt = str(package_dir / "test" / "test.pptx")
        self._models: ModelManager | None = None
        self.config = Config(self.template)

    @property
    def models(self) -> ModelManager:
        if self._models is None:
            self._models = ModelManager()
        return self._models

    def get_slide_induction(self) -> dict:
        with open(Path(self.template) / "slide_induction.json", encoding="utf-8") as f:
            return json.load(f)

    def get_document_json(self) -> dict:
        path = Path(self.document) / "refined_doc.json"
        if not path.exists():
            pytest.skip(f"legacy document fixture is unavailable: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def get_image_stats(self) -> dict:
        with open(Path(self.template) / "image_stats.json", encoding="utf-8") as f:
            return json.load(f)

    @property
    def language_model(self):
        return self.models.language_model

    @property
    def vision_model(self):
        return self.models.vision_model

    @property
    def image_model(self):
        return self.models.image_model


test_config = TestConfig()
