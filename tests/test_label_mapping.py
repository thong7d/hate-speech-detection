import pytest

from src.models.registry import build_label_mapping, validate_label_mapping


def test_label_mapping_has_required_keys() -> None:
    mapping = build_label_mapping(["CLEAN", "OFFENSIVE", "HATE"])
    assert set(mapping) == {"id2label", "label2id"}
    assert mapping["id2label"]["0"] == "CLEAN"
    assert mapping["label2id"]["HATE"] == 2


def test_label_mapping_is_bidirectional() -> None:
    mapping = build_label_mapping(["CLEAN", "OFFENSIVE", "HATE"])
    validate_label_mapping(mapping)


def test_inconsistent_label_mapping_fails() -> None:
    with pytest.raises(ValueError):
        validate_label_mapping({"id2label": {"0": "CLEAN"}, "label2id": {"CLEAN": 1}})
