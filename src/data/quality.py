"""
Data quality checks for ViHSD processed splits.

Run from the repo root:
    python -m src.data.quality
or, when PYTHONPATH=src:
    python -m data.quality
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

try:
    from data.preprocessing import clean_text, normalize_text_label_frame, resolve_project_path
except Exception:  # package imported as src.data.* or optional deps missing
    try:
        from src.data.preprocessing import clean_text, normalize_text_label_frame, resolve_project_path
    except Exception:
        def clean_text(text):
            return "" if text is None else " ".join(str(text).split())

        def normalize_text_label_frame(df):
            raise RuntimeError("pandas is required for data quality checks")

        def resolve_project_path(template_path, project_root="."):
            return Path(str(template_path).replace("{project_root}", str(project_root))).resolve()


DEFAULT_LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def _read_split(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file type: {path}")


def _split_summary(name: str, df: pd.DataFrame, label_map: Mapping[int, str]) -> dict:
    canonical = normalize_text_label_frame(df)
    normalized_text = canonical["text"].map(clean_text)
    counts = canonical["label"].value_counts().sort_index().to_dict()
    duplicate_count = int(normalized_text.duplicated().sum())
    return {
        "name": name,
        "rows": int(len(canonical)),
        "classes": {label_map.get(int(k), str(k)): int(v) for k, v in counts.items()},
        "duplicates": duplicate_count,
        "texts": set(normalized_text),
    }


def build_data_quality_report(
    split_paths: Mapping[str, Path],
    *,
    label_map: Mapping[int, str] | None = None,
) -> str:
    """Build a markdown report covering distribution, duplicates, and split overlap."""
    label_map = label_map or DEFAULT_LABEL_MAP
    loaded: dict[str, dict] = {}
    missing: list[str] = []

    for name, path in split_paths.items():
        if not path.exists():
            missing.append(f"{name}: {path}")
            continue
        loaded[name] = _split_summary(name, _read_split(path), label_map)

    lines = [
        "# Data Quality Report",
        "",
        "## Status",
    ]
    if missing:
        lines.append("Khong du du lieu de xac minh tat ca split.")
        lines.append("")
        lines.append("Missing files:")
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("All configured splits were found.")

    lines.extend(["", "## Class Distribution", "", "| Split | Rows | Class counts | Duplicate texts |", "|---|---:|---|---:|"])
    for name in split_paths:
        info = loaded.get(name)
        if info is None:
            lines.append(f"| {name} | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {name} | {info['rows']} | {info['classes']} | {info['duplicates']} |"
        )

    lines.extend(["", "## Split Overlap", "", "| Pair | Overlap count |", "|---|---:|"])
    names = list(loaded)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = loaded[left]["texts"] & loaded[right]["texts"]
            lines.append(f"| {left} / {right} | {len(overlap)} |")

    if len(loaded) < 2:
        lines.append("| N/A | Khong du du lieu de xac minh |")

    lines.extend(
        [
            "",
            "## Notes",
            "- Class weights must be computed from the train split only.",
            "- Duplicate and overlap checks use the shared clean_text() normalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(config_path: str = "configs/paths.yaml", output_path: str = "results/data_quality_report.md") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if pd is None or OmegaConf is None:
        missing = []
        if pd is None:
            missing.append("pandas")
        if OmegaConf is None:
            missing.append("omegaconf")
        out.write_text(
            "# Data Quality Report\n\n"
            "Khong du du lieu de xac minh trong moi truong hien tai.\n\n"
            f"Missing Python dependencies: {', '.join(missing)}\n",
            encoding="utf-8",
        )
        print(f"Data quality report written to {out}")
        return

    cfg = OmegaConf.load(config_path)
    project_root = Path(cfg.get("project_root", ".")).resolve()

    split_paths = {
        "train": resolve_project_path(cfg.data.train_processed, project_root),
        "val": resolve_project_path(cfg.data.val_processed, project_root),
        "test": resolve_project_path(cfg.data.test_processed, project_root),
    }
    label_map = {int(k): str(v) for k, v in dict(cfg.model.label_map).items()}
    report = build_data_quality_report(split_paths, label_map=label_map)

    out = resolve_project_path(output_path, project_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Data quality report written to {out}")


if __name__ == "__main__":
    main()
