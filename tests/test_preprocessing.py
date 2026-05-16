from src.data.preprocessing import clean_text, preprocess_text


def test_preprocessing_handles_empty_input() -> None:
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_preprocessing_returns_string() -> None:
    assert isinstance(preprocess_text("  hello   world  "), str)
    assert preprocess_text("  hello   world  ") == "hello world"


def test_training_and_inference_share_preprocessing_alias() -> None:
    assert preprocess_text("visit http://example.com now") == clean_text("visit http://example.com now")
