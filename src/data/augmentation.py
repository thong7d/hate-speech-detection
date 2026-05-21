"""
Data augmentation utilities for Vietnamese hate speech detection.

Techniques implemented:
  1. EDA (Easy Data Augmentation) — random deletion, random swap
  2. Diacritic removal — create no-accent copies of toxic texts
  3. Teencode variant generation — convert standard text to teencode
"""
from __future__ import annotations

import random
import re
import unicodedata
from typing import Mapping

import pandas as pd


# ============================================================
# Vietnamese diacritic removal table
# ============================================================

_DIACRITIC_MAP = str.maketrans(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)

# Reverse teencode: convert standard Vietnamese back to common teencode
_REVERSE_TEENCODE: dict[str, list[str]] = {
    "không": ["ko", "k", "khg"],
    "được": ["dc", "dk"],
    "mình": ["mk", "mik"],
    "bạn": ["bn", "b"],
    "bây giờ": ["bjo", "bh"],
    "bình thường": ["bt", "bth"],
    "cũng": ["cx", "cg"],
    "rồi": ["r", "rr"],
    "với": ["vs"],
    "mới": ["ms"],
    "mọi người": ["mn", "mng"],
    "người": ["ng"],
    "người ta": ["ngta"],
    "gì": ["j", "z", "g"],
    "trước": ["trc"],
    "luôn": ["lun"],
    "làm": ["lm"],
    "đang": ["dg", "dag"],
    "hiểu": ["hiu"],
    "quá": ["qá", "wa"],
}


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics from text."""
    return text.translate(_DIACRITIC_MAP)


def generate_teencode_variant(text: str, probability: float = 0.3) -> str:
    """Convert some standard Vietnamese words to teencode variants.

    Args:
        text: Input text with standard Vietnamese.
        probability: Probability of converting each eligible word.

    Returns:
        Text with some words replaced by teencode variants.
    """
    words = text.split()
    result = []
    for word in words:
        lower_word = word.lower()
        if lower_word in _REVERSE_TEENCODE and random.random() < probability:
            replacement = random.choice(_REVERSE_TEENCODE[lower_word])
            result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


# ============================================================
# EDA (Easy Data Augmentation) for Vietnamese
# ============================================================

def eda_random_deletion(text: str, alpha: float = 0.15) -> str:
    """Randomly delete words with probability alpha.

    Args:
        text: Input text.
        alpha: Probability of deleting each word.

    Returns:
        Text with some words randomly deleted.
    """
    words = text.split()
    if len(words) <= 2:
        return text
    remaining = [w for w in words if random.random() > alpha]
    if not remaining:
        return random.choice(words)
    return " ".join(remaining)


def eda_random_swap(text: str, n_swaps: int = 1) -> str:
    """Randomly swap positions of n pairs of words.

    Args:
        text: Input text.
        n_swaps: Number of word pairs to swap.

    Returns:
        Text with word positions swapped.
    """
    words = text.split()
    if len(words) < 2:
        return text
    words = list(words)
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def eda_augment_text(
    text: str,
    alpha_rd: float = 0.15,
    alpha_rs: float = 0.1,
) -> str:
    """Apply one random EDA technique to the text.

    Args:
        text: Input text.
        alpha_rd: Probability for random deletion.
        alpha_rs: Probability used to compute swaps.

    Returns:
        Augmented text.
    """
    technique = random.choice(["deletion", "swap"])
    if technique == "deletion":
        return eda_random_deletion(text, alpha=alpha_rd)
    else:
        n_swaps = max(1, int(alpha_rs * len(text.split())))
        return eda_random_swap(text, n_swaps=n_swaps)


# ============================================================
# DataFrame-level augmentation
# ============================================================

def augment_with_eda(
    df: pd.DataFrame,
    label2id: Mapping[str, int],
    *,
    target_labels: list[str] | None = None,
    alpha_rd: float = 0.15,
    alpha_rs: float = 0.1,
    num_aug_per_sample: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Apply EDA augmentation to minority class samples.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        label2id: Mapping from label name to label index.
        target_labels: Labels to augment (default: OFFENSIVE, HATE).
        alpha_rd: Random deletion probability.
        alpha_rs: Random swap probability factor.
        num_aug_per_sample: Number of augmented copies per original sample.
        seed: Random seed.

    Returns:
        Tuple of (augmented DataFrame, summary dict).
    """
    random.seed(seed)
    if target_labels is None:
        target_labels = ["OFFENSIVE", "HATE"]

    target_ids = {int(label2id[label]) for label in target_labels if label in label2id}
    augmented_rows: list[dict] = []

    for _, row in df.iterrows():
        if int(row["label"]) not in target_ids:
            continue
        for _ in range(num_aug_per_sample):
            aug_text = eda_augment_text(
                str(row["text"]),
                alpha_rd=alpha_rd,
                alpha_rs=alpha_rs,
            )
            if aug_text and len(aug_text.strip()) >= 3:
                augmented_rows.append({"text": aug_text, "label": int(row["label"])})

    if not augmented_rows:
        return df, {"enabled": True, "added_examples": 0}

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    summary = {
        "enabled": True,
        "added_examples": len(augmented_rows),
        "target_labels": target_labels,
        "num_aug_per_sample": num_aug_per_sample,
        "alpha_rd": alpha_rd,
        "alpha_rs": alpha_rs,
    }
    return combined, summary


def augment_with_diacritic_removal(
    df: pd.DataFrame,
    label2id: Mapping[str, int],
    *,
    target_labels: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Create diacritic-removed copies of toxic texts.

    This helps the model learn that text without Vietnamese accents
    can still be toxic (common in informal online communication).

    Args:
        df: DataFrame with 'text' and 'label' columns.
        label2id: Mapping from label name to label index.
        target_labels: Labels to augment (default: OFFENSIVE, HATE).
        seed: Random seed.

    Returns:
        Tuple of (augmented DataFrame, summary dict).
    """
    if target_labels is None:
        target_labels = ["OFFENSIVE", "HATE"]

    target_ids = {int(label2id[label]) for label in target_labels if label in label2id}
    augmented_rows: list[dict] = []

    for _, row in df.iterrows():
        if int(row["label"]) not in target_ids:
            continue
        no_accent = remove_diacritics(str(row["text"]))
        if no_accent != str(row["text"]) and len(no_accent.strip()) >= 3:
            augmented_rows.append({"text": no_accent, "label": int(row["label"])})

    if not augmented_rows:
        return df, {"enabled": True, "added_examples": 0}

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined, {
        "enabled": True,
        "added_examples": len(augmented_rows),
        "target_labels": target_labels,
    }


def augment_with_teencode(
    df: pd.DataFrame,
    label2id: Mapping[str, int],
    *,
    target_labels: list[str] | None = None,
    probability: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Create teencode variant copies of toxic texts.

    This helps the model learn that teencode/internet slang versions
    of offensive text are still toxic.

    Args:
        df: DataFrame with 'text' and 'label' columns.
        label2id: Mapping from label name to label index.
        target_labels: Labels to augment (default: OFFENSIVE, HATE).
        probability: Probability of converting each word to teencode.
        seed: Random seed.

    Returns:
        Tuple of (augmented DataFrame, summary dict).
    """
    random.seed(seed)
    if target_labels is None:
        target_labels = ["OFFENSIVE", "HATE"]

    target_ids = {int(label2id[label]) for label in target_labels if label in label2id}
    augmented_rows: list[dict] = []

    for _, row in df.iterrows():
        if int(row["label"]) not in target_ids:
            continue
        teencode_text = generate_teencode_variant(str(row["text"]), probability=probability)
        if teencode_text != str(row["text"]) and len(teencode_text.strip()) >= 3:
            augmented_rows.append({"text": teencode_text, "label": int(row["label"])})

    if not augmented_rows:
        return df, {"enabled": True, "added_examples": 0}

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return combined, {
        "enabled": True,
        "added_examples": len(augmented_rows),
        "target_labels": target_labels,
        "probability": probability,
    }
