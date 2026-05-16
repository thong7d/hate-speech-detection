from __future__ import annotations

import argparse
import json

try:
    from src.training.trainer import train_from_config
    from src.utils.config import load_yaml_config
except ImportError:
    from training.trainer import train_from_config
    from utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hate speech classifier.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    result = train_from_config(load_yaml_config(args.config))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
