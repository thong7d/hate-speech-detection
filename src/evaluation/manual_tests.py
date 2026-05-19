from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from src.models.classifier import HateSpeechClassifier
    from src.training.robustness_cases import (
        CONTRASTIVE_HOLDOUT_CASES,
        CONTRASTIVE_TRAIN_CASES,
        DIACRITIC_HOLDOUT_CASES,
        DIACRITIC_TRAIN_CASES,
        ROBUSTNESS_HOLDOUT_CASES,
        ROBUSTNESS_TRAIN_CASES,
        normalized_case_texts,
        summarize_cases,
    )
    from src.utils.config import load_yaml_config
except ImportError:
    from models.classifier import HateSpeechClassifier
    from training.robustness_cases import (
        CONTRASTIVE_HOLDOUT_CASES,
        CONTRASTIVE_TRAIN_CASES,
        DIACRITIC_HOLDOUT_CASES,
        DIACRITIC_TRAIN_CASES,
        ROBUSTNESS_HOLDOUT_CASES,
        ROBUSTNESS_TRAIN_CASES,
        normalized_case_texts,
        summarize_cases,
    )
    from utils.config import load_yaml_config


TEST_CASES = ROBUSTNESS_HOLDOUT_CASES + CONTRASTIVE_HOLDOUT_CASES + DIACRITIC_HOLDOUT_CASES


def run_manual_tests(config_path: str = "configs/model.yaml", output_path: str = "results/manual_test_report.md") -> list[dict]:
    config = load_yaml_config(config_path)
    rows: list[dict] = []
    classifier = None
    load_error = None
    try:
        classifier = HateSpeechClassifier.from_config(config)
    except Exception as exc:
        load_error = str(exc)

    for text, expected, category in TEST_CASES:
        if classifier is None:
            rows.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": "N/A",
                    "confidence": "N/A",
                    "probabilities": "N/A",
                    "pass": "N/A",
                    "category": category,
                }
            )
            continue
        pred = classifier.predict(text)
        predicted = pred["label"]
        rows.append(
            {
                "text": text,
                "expected": expected,
                "predicted": predicted,
                "confidence": f"{pred['confidence']:.4f}",
                "probabilities": json.dumps(pred["probabilities"], ensure_ascii=False, sort_keys=True),
                "pass": "yes" if predicted == expected else "no",
                "category": category,
            }
        )

    report = build_markdown_report(rows, load_error=load_error)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return rows


def build_markdown_report(rows: list[dict], *, load_error: str | None = None) -> str:
    lines = ["# Manual Robustness Test Report", ""]
    overlap = _training_overlap(TEST_CASES)
    lines.extend(
        [
            "## Summary",
            "",
            f"- Total cases: {len(rows)}",
            f"- Passed cases: {sum(1 for row in rows if row['pass'] == 'yes')}",
            f"- Exact overlap with train augmentation: {len(overlap)}",
            "",
        ]
    )
    lines.extend(_summary_lines(rows))
    lines.append("")
    if overlap:
        lines.extend(["## Overlap Warning", ""])
        lines.extend(f"- {text}" for text in overlap)
        lines.append("")
    if load_error:
        lines.extend(["Khong du du lieu de xac minh", "", f"Model load error: `{load_error}`", ""])
    lines.extend(
        [
            "| text | expected | predicted | confidence | probabilities | pass | category |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {text} | {expected} | {predicted} | {confidence} | {probabilities} | {pass_} | {category} |".format(
                text=row["text"].replace("|", "\\|"),
                expected=row["expected"],
                predicted=row["predicted"],
                confidence=row["confidence"],
                probabilities=str(row["probabilities"]).replace("|", "\\|"),
                pass_=row["pass"],
                category=row["category"],
            )
        )
    return "\n".join(lines) + "\n"


def _summary_lines(rows: list[dict]) -> list[str]:
    case_summary = summarize_cases(TEST_CASES)
    lines = [
        "### Case Distribution",
        "",
        "| Label | Cases |",
        "|---|---:|",
    ]
    for label, count in sorted(case_summary["label_counts"].items()):
        lines.append(f"| {label} | {count} |")

    lines.extend(["", "### Category Accuracy", "", "| Category | Passed | Total | Accuracy |", "|---|---:|---:|---:|"])
    categories = sorted({row["category"] for row in rows})
    for category in categories:
        category_rows = [row for row in rows if row["category"] == category]
        passed = sum(1 for row in category_rows if row["pass"] == "yes")
        total = len(category_rows)
        accuracy = passed / total if total else 0
        lines.append(f"| {category} | {passed} | {total} | {accuracy:.2%} |")
    return lines


def _training_overlap(cases: list[tuple[str, str, str]]) -> list[str]:
    train_texts = normalized_case_texts(ROBUSTNESS_TRAIN_CASES + CONTRASTIVE_TRAIN_CASES + DIACRITIC_TRAIN_CASES)
    overlap = []
    for text, _label, _category in cases:
        normalized = " ".join(text.lower().split())
        if normalized in train_texts:
            overlap.append(text)
    return overlap


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manual robustness prompts against the model.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--output", default="results/manual_test_report.md")
    args = parser.parse_args()
    rows = run_manual_tests(args.config, args.output)
    print(f"Manual test rows: {len(rows)}")
    print(f"Report written to {Path(args.output)}")


if __name__ == "__main__":
    main()
