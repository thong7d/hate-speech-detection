from __future__ import annotations

import torch

# PyTorch 2.6+ weights_only=True compatibility guard for NumPy 1.x/2.x RNG states
try:
    import numpy as np
    safe_globals = []
    
    # 1. Resolve core array reconstruction functions
    if hasattr(np, "_core") and hasattr(np._core, "multiarray") and hasattr(np._core.multiarray, "_reconstruct"):
        safe_globals.append(np._core.multiarray._reconstruct)
    if hasattr(np, "core") and hasattr(np.core, "multiarray") and hasattr(np.core.multiarray, "_reconstruct"):
        safe_globals.append(np.core.multiarray._reconstruct)
        
    # 2. Resolve standard NumPy data types
    if hasattr(np, "dtype"):
        safe_globals.append(np.dtype)
    if hasattr(np, "ndarray"):
        safe_globals.append(np.ndarray)
        
    # 3. Resolve NumPy 2.0+ specific strict DTypes required for RNG unpickling
    try:
        import numpy.dtypes
        if hasattr(numpy.dtypes, "UInt32DType"):
            safe_globals.append(numpy.dtypes.UInt32DType)
    except ImportError:
        pass

    # Register all resolved globals into PyTorch trusted serialization sandbox
    if safe_globals and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_globals)
except (ImportError, AttributeError):
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
