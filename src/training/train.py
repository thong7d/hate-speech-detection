from __future__ import annotations

import torch
try:
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        import numpy
        # Allowlist standard NumPy classes and reconstruction globals for safe loading in PyTorch 2.6+
        safe_numpy_globals = []
        for name in [
            "numpy._core.multiarray._reconstruct",
            "numpy.core.multiarray._reconstruct",
            "numpy.dtype",
            "numpy.ndarray"
        ]:
            try:
                parts = name.split(".")
                obj = __import__(parts[0])
                for part in parts[1:]:
                    obj = getattr(obj, part)
                safe_numpy_globals.append(obj)
            except AttributeError:
                pass
        if safe_numpy_globals:
            torch.serialization.add_safe_globals(safe_numpy_globals)
except Exception:
    pass

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
