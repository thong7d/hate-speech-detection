from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from src.models.classifier import HateSpeechClassifier
    from src.utils.config import load_yaml_config
except ImportError:
    from models.classifier import HateSpeechClassifier
    from utils.config import load_yaml_config


UNKNOWN_EXPECTED = "Khong du du lieu de xac minh expected label"

TEST_CASES = [
    ("I hate this group.", "subject_change"),
    ("You hate this group.", "subject_change"),
    ("They hate this group.", "subject_change"),
    ("He hates this group.", "subject_change"),
    ("She hates this group.", "subject_change"),
    ("People hate this group.", "subject_change"),
    ("This article discusses hate speech.", "keyword_context"),
    ("The word is offensive, but I am quoting it for analysis.", "keyword_context"),
    ("He explained why discrimination is harmful.", "keyword_context"),
    ("The dataset contains toxic words.", "keyword_context"),
    ("We should stop racism and hate speech.", "keyword_context"),
    ("I do not hate anyone.", "negation"),
    ("We should not attack people because of their identity.", "negation"),
    ("No group deserves discrimination.", "negation"),
    ("Hate speech should be prevented.", "negation"),
    ("Those people should not be allowed here.", "implicit_hate"),
    ("They do not belong in our country.", "implicit_hate"),
    ("I went to school today.", "neutral"),
]


def run_manual_tests(config_path: str = "configs/model.yaml", output_path: str = "results/manual_test_report.md") -> list[dict]:
    config = load_yaml_config(config_path)
    rows: list[dict] = []
    classifier = None
    load_error = None
    try:
        classifier = HateSpeechClassifier.from_config(config)
    except Exception as exc:
        load_error = str(exc)

    for text, category in TEST_CASES:
        if classifier is None:
            rows.append(
                {
                    "text": text,
                    "expected": UNKNOWN_EXPECTED,
                    "predicted": "N/A",
                    "confidence": "N/A",
                    "probabilities": "N/A",
                    "category": category,
                }
            )
            continue
        pred = classifier.predict(text)
        rows.append(
            {
                "text": text,
                "expected": UNKNOWN_EXPECTED,
                "predicted": pred["label"],
                "confidence": f"{pred['confidence']:.4f}",
                "probabilities": json.dumps(pred["probabilities"], ensure_ascii=False, sort_keys=True),
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
    if load_error:
        lines.extend(["Khong du du lieu de xac minh", "", f"Model load error: `{load_error}`", ""])
    lines.extend(["| text | expected | predicted | confidence | probabilities | category |", "|---|---|---|---:|---|---|"])
    for row in rows:
        lines.append(
            "| {text} | {expected} | {predicted} | {confidence} | {probabilities} | {category} |".format(
                text=row["text"].replace("|", "\\|"),
                expected=row["expected"],
                predicted=row["predicted"],
                confidence=row["confidence"],
                probabilities=str(row["probabilities"]).replace("|", "\\|"),
                category=row["category"],
            )
        )
    return "\n".join(lines) + "\n"


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
