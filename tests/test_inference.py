from src.models.classifier import HateSpeechClassifier


class DummyClassifier(HateSpeechClassifier):
    def __init__(self) -> None:
        pass

    @property
    def model_version(self) -> str:
        return "v-test"

    def predict(self, text: str) -> dict:
        return {
            "text": text,
            "label": "CLEAN",
            "confidence": 0.99,
            "probabilities": {"CLEAN": 0.99, "OFFENSIVE": 0.01, "HATE": 0.0},
            "model_version": "v-test",
        }


def test_predict_returns_required_fields() -> None:
    result = DummyClassifier().predict("")
    assert {"text", "label", "confidence", "probabilities", "model_version"} <= set(result)


def test_predict_batch_returns_list() -> None:
    results = DummyClassifier().predict_batch(["a", "b"])
    assert isinstance(results, list)
    assert len(results) == 2
