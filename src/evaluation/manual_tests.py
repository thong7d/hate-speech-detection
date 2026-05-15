"""
Manual robustness tests for subject changes, negation, quoting/context,
keyword bias, and neutral sentences.

Run from the repo root:
    python -m src.evaluation.manual_tests --model-source thong7d/vihsd-xlmr-hate-speech
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


TEST_CASES = [
    # Subject-change stability
    ("I hate this group.", "HATE", "subject_change"),
    ("You hate this group.", "HATE", "subject_change"),
    ("They hate this group.", "HATE", "subject_change"),
    ("He hates this group.", "HATE", "subject_change"),
    ("She hates this group.", "HATE", "subject_change"),
    ("People hate this group.", "HATE", "subject_change"),
    ("A person hates this group.", "HATE", "subject_change"),
    # Sensitive keywords in non-hate context
    ("This article discusses hate speech.", "CLEAN", "keyword_context"),
    ("The word is offensive, but I am quoting it for analysis.", "CLEAN", "keyword_context"),
    ("He explained why discrimination is harmful.", "CLEAN", "keyword_context"),
    ("The dataset contains toxic words.", "CLEAN", "keyword_context"),
    ("We should stop racism and hate speech.", "CLEAN", "keyword_context"),
    ("The teacher gave a lecture about online abuse.", "CLEAN", "keyword_context"),
    # Implicit hate
    ("Those people should not be allowed here.", "HATE", "implicit_hate"),
    ("They do not belong in our country.", "HATE", "implicit_hate"),
    ("Nobody from that group should get a job.", "HATE", "implicit_hate"),
    ("We should keep them away from our schools.", "HATE", "implicit_hate"),
    ("People like them should be excluded.", "HATE", "implicit_hate"),
    # Neutral
    ("I went to school today.", "CLEAN", "neutral"),
    ("The weather is nice.", "CLEAN", "neutral"),
    ("She is reading a book.", "CLEAN", "neutral"),
    ("They are playing football.", "CLEAN", "neutral"),
    ("This is a normal sentence.", "CLEAN", "neutral"),
    ("The meeting starts at 9 AM.", "CLEAN", "neutral"),
    # Negation
    ("I do not hate anyone.", "CLEAN", "negation"),
    ("We should not attack people because of their identity.", "CLEAN", "negation"),
    ("No group deserves discrimination.", "CLEAN", "negation"),
    ("Hate speech should be prevented.", "CLEAN", "negation"),
    ("I disagree with racism.", "CLEAN", "negation"),
    ("People should be treated equally.", "CLEAN", "negation"),
]


def _probabilities_to_text(probabilities: dict[str, float]) -> str:
    return json.dumps(probabilities, ensure_ascii=False, sort_keys=True)


def build_markdown_report(rows: list[dict], *, load_error: str | None = None) -> str:
    lines = ["# Manual Robustness Test Report", ""]
    if load_error:
        lines.extend(
            [
                "Khong du du lieu de xac minh ket qua chay trong moi truong hien tai.",
                "",
                f"Model load error: `{load_error}`",
                "",
            ]
        )

    lines.extend(
        [
            "| text | expected | predicted | confidence | probabilities | pass | category |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {text} | {expected} | {predicted} | {confidence} | {probabilities} | {passed} | {category} |".format(
                text=row["text"].replace("|", "\\|"),
                expected=row["expected"],
                predicted=row["predicted"],
                confidence=row["confidence"],
                probabilities=row["probabilities"].replace("|", "\\|"),
                passed=row["pass"],
                category=row["category"],
            )
        )
    return "\n".join(lines) + "\n"


def run_manual_tests(
    model_source: str,
    *,
    output_path: str | Path = "results/manual_test_report.md",
    max_length: int = 128,
) -> list[dict]:
    rows: list[dict] = []
    load_error: str | None = None
    artifacts = None
    predict_with_artifacts = None

    try:
        try:
            from evaluation.classifier import load_hf_artifacts, predict_with_artifacts as _predict
        except ImportError:
            from src.evaluation.classifier import load_hf_artifacts, predict_with_artifacts as _predict
        predict_with_artifacts = _predict
        artifacts = load_hf_artifacts(
            model_source,
            token=os.environ.get("HF_TOKEN"),
            device="auto",
            max_length=max_length,
        )
    except Exception as exc:
        load_error = str(exc)

    for text, expected, category in TEST_CASES:
        if artifacts is None:
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

        pred = predict_with_artifacts(artifacts, text, preprocess=True)
        rows.append(
            {
                "text": text,
                "expected": expected,
                "predicted": pred["label"],
                "confidence": f"{pred['confidence']:.4f}",
                "probabilities": _probabilities_to_text(pred["probabilities"]),
                "pass": "PASS" if pred["label"] == expected else "FAIL",
                "category": category,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_report(rows, load_error=load_error), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-source",
        default=os.environ.get("HF_MODEL_ID", "thong7d/vihsd-xlmr-hate-speech"),
    )
    parser.add_argument("--output", default="results/manual_test_report.md")
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "128")))
    args = parser.parse_args()

    rows = run_manual_tests(args.model_source, output_path=args.output, max_length=args.max_length)
    print(f"Manual test rows: {len(rows)}")
    print(f"Report written to {Path(args.output)}")


if __name__ == "__main__":
    main()
