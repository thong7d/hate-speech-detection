from fastapi.testclient import TestClient

from src.api import app as app_module


class FakeClassifier:
    metadata = {"model_version": "v-test"}

    def predict(self, text: str) -> dict:
        return {
            "text": text,
            "label": "CLEAN",
            "confidence": 0.99,
            "probabilities": {"CLEAN": 0.99, "OFFENSIVE": 0.01, "HATE": 0.0},
            "model_version": "v-test",
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]


def test_health() -> None:
    client = TestClient(app_module.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_schema() -> None:
    app_module.app.dependency_overrides[app_module.get_classifier] = lambda: FakeClassifier()
    client = TestClient(app_module.app)
    response = client.post("/predict", json={"text": "hello"})
    app_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["label"] == "CLEAN"


def test_predict_batch_schema() -> None:
    app_module.app.dependency_overrides[app_module.get_classifier] = lambda: FakeClassifier()
    client = TestClient(app_module.app)
    response = client.post("/predict-batch", json={"texts": ["a", "b"]})
    app_module.app.dependency_overrides.clear()
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2
