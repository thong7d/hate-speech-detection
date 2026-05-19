from src.training.robustness_cases import (
    CONTRASTIVE_HOLDOUT_CASES,
    CONTRASTIVE_TRAIN_CASES,
    DIACRITIC_HOLDOUT_CASES,
    DIACRITIC_TRAIN_CASES,
    ROBUSTNESS_HOLDOUT_CASES,
    ROBUSTNESS_TRAIN_CASES,
    normalized_case_texts,
)


def test_train_and_holdout_cases_do_not_overlap() -> None:
    train_cases = ROBUSTNESS_TRAIN_CASES + CONTRASTIVE_TRAIN_CASES + DIACRITIC_TRAIN_CASES
    holdout_cases = ROBUSTNESS_HOLDOUT_CASES + CONTRASTIVE_HOLDOUT_CASES + DIACRITIC_HOLDOUT_CASES

    assert normalized_case_texts(train_cases).isdisjoint(normalized_case_texts(holdout_cases))


def test_contrastive_cases_cover_all_labels() -> None:
    labels = {label for _text, label, _category in CONTRASTIVE_TRAIN_CASES}

    assert {"CLEAN", "OFFENSIVE", "HATE"} <= labels


def test_diacritic_cases_cover_all_labels() -> None:
    labels = {label for _text, label, _category in DIACRITIC_TRAIN_CASES}

    assert {"CLEAN", "OFFENSIVE", "HATE"} <= labels
