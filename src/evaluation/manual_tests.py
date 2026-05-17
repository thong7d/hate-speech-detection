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


TEST_CASES = [
    ("Hom nay troi dep, toi di hoc cung ban.", "CLEAN", "neutral"),
    ("Bai viet nay phan tich hien tuong ky thi tren mang.", "CLEAN", "neutral"),
    ("Chung ta nen ton trong su khac biet cua moi nguoi.", "CLEAN", "neutral"),
    ("Toi khong ghet ai ca.", "CLEAN", "negation"),
    ("Khong ai dang bi xuc pham trong cau nay.", "CLEAN", "negation"),
    ("Khong nen cong kich nguoi khac vi gioi tinh hay que quan.", "CLEAN", "negation"),
    ("Can ngan chan loi noi thu han tren mang xa hoi.", "CLEAN", "keyword_context"),
    ("Tu ngu do rat xuc pham, nhung toi dang trich dan de phan tich.", "CLEAN", "keyword_context"),
    ("Bao cao nay noi ve viec phat hien ngon tu thu ghet.", "CLEAN", "keyword_context"),
    ("May noi chuyen ngu that.", "OFFENSIVE", "direct_insult"),
    ("Thang nay dung la do vo hoc.", "OFFENSIVE", "direct_insult"),
    ("Con nho do noi nang mat day.", "OFFENSIVE", "direct_insult"),
    ("Bai viet cua may nham nhi va rac ruoi.", "OFFENSIVE", "direct_insult"),
    ("Lu do khong nen duoc song o day.", "HATE", "group_attack"),
    ("Duoi het bon do ra khoi dat nuoc nay.", "HATE", "group_attack"),
    ("Nhung nguoi nhu vay khong xung dang co quyen loi.", "HATE", "group_attack"),
    ("Cam cho cai nhom do di hoc va di lam.", "HATE", "group_attack"),
    ("Tui no la mam benh cua xa hoi.", "HATE", "dehumanization"),
    ("Loai nguoi do chi lam ban cong dong.", "HATE", "dehumanization"),
    ("Khong cho bon do xuat hien o khu nay nua.", "HATE", "exclusion"),
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
