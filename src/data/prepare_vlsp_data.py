import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(
    vlsp_dir: str,
    vihsd_train_path: str,
    vihsd_dev_path: str,
    output_train_path: str,
    output_dev_path: str,
    rehearsal_size: int = 2500,
    dev_size: int = 500,
    seed: int = 42
):
    print(f"[PREPARE] VLSP raw directory: {vlsp_dir}")
    print(f"[PREPARE] ViHSD train path: {vihsd_train_path}")
    print(f"[PREPARE] ViHSD dev path: {vihsd_dev_path}")

    # 1. Load and merge VLSP-2019 raw files
    vlsp_dir_path = Path(vlsp_dir)
    
    # Path resolution fallback for Google Drive mounting variations
    if not (vlsp_dir_path / "02_train_text.csv").exists() or not (vlsp_dir_path / "03_train_label.csv").exists():
        str_path = str(vlsp_dir)
        if "MyDrive" in str_path:
            fallback_str = str_path.replace("MyDrive", "My Drive")
            fallback_path = Path(fallback_str)
            if (fallback_path / "02_train_text.csv").exists() and (fallback_path / "03_train_label.csv").exists():
                print(f"[PREPARE] Fallback: Using '{fallback_str}' instead of '{str_path}'")
                vlsp_dir_path = fallback_path
        elif "My Drive" in str_path:
            fallback_str = str_path.replace("My Drive", "MyDrive")
            fallback_path = Path(fallback_str)
            if (fallback_path / "02_train_text.csv").exists() and (fallback_path / "03_train_label.csv").exists():
                print(f"[PREPARE] Fallback: Using '{fallback_str}' instead of '{str_path}'")
                vlsp_dir_path = fallback_path

    text_path = vlsp_dir_path / "02_train_text.csv"
    label_path = vlsp_dir_path / "03_train_label.csv"

    if not text_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"VLSP raw data files not found in {vlsp_dir} (or tried fallback). "
            "Please ensure you uploaded '02_train_text.csv' and '03_train_label.csv' "
            "to your Google Drive folder."
        )

    df_vlsp_text = pd.read_csv(text_path)
    df_vlsp_label = pd.read_csv(label_path)
    df_vlsp = pd.merge(df_vlsp_text, df_vlsp_label, on="id")

    # Rename columns to standard format
    df_vlsp = df_vlsp.rename(columns={"free_text": "text", "label_id": "label"})
    df_vlsp = df_vlsp[["text", "label"]].dropna()
    df_vlsp["label"] = df_vlsp["label"].astype(int)

    # Split VLSP-2019 into train and dev (90/10 stratified)
    df_vlsp_train, df_vlsp_dev = train_test_split(
        df_vlsp,
        test_size=0.1,
        stratify=df_vlsp["label"],
        random_state=seed
    )

    print(f"[PREPARE] VLSP split: Train={len(df_vlsp_train)}, Dev={len(df_vlsp_dev)}")

    # 2. Load ViHSD train dataset for Rehearsal Buffer
    df_vihsd_train = pd.read_parquet(vihsd_train_path)
    df_vihsd_train = df_vihsd_train[["text", "label"]].dropna()
    df_vihsd_train["label"] = df_vihsd_train["label"].astype(int)

    # Stratified sampling for Rehearsal Buffer
    df_rehearsal, _ = train_test_split(
        df_vihsd_train,
        train_size=min(rehearsal_size, len(df_vihsd_train)),
        stratify=df_vihsd_train["label"],
        random_state=seed
    )
    print(f"[PREPARE] Sampled ViHSD Rehearsal Buffer size: {len(df_rehearsal)}")

    # 3. Combine VLSP train & ViHSD rehearsal
    df_vlsp_train = df_vlsp_train.copy()
    df_rehearsal = df_rehearsal.copy()
    df_vlsp_train["source"] = "vlsp"
    df_rehearsal["source"] = "vihsd"

    df_continual_train = pd.concat([df_vlsp_train, df_rehearsal], ignore_index=True)

    # 4. Calculate sampling weights for PyTorch WeightedRandomSampler
    # Target batch ratio: 25% ViHSD (rehearsal) and 75% VLSP (new data)
    n_vihsd = len(df_rehearsal)
    n_vlsp = len(df_vlsp_train)

    w_vihsd = 0.25 / n_vihsd if n_vihsd > 0 else 0.0
    w_vlsp = 0.75 / n_vlsp if n_vlsp > 0 else 0.0

    df_continual_train["sample_weight"] = df_continual_train["source"].map(
        {"vihsd": w_vihsd, "vlsp": w_vlsp}
    )

    # 5. Create joint dev validation set (50/50 mix to solve Validation Strategy Bias)
    df_vihsd_dev = pd.read_parquet(vihsd_dev_path)
    df_vihsd_dev = df_vihsd_dev[["text", "label"]].dropna()
    df_vihsd_dev["label"] = df_vihsd_dev["label"].astype(int)

    # Stratified sampling of ViHSD dev
    df_vihsd_dev_sampled, _ = train_test_split(
        df_vihsd_dev,
        train_size=min(dev_size, len(df_vihsd_dev)),
        stratify=df_vihsd_dev["label"],
        random_state=seed
    )

    # Stratified sampling of VLSP dev
    df_vlsp_dev_sampled, _ = train_test_split(
        df_vlsp_dev,
        train_size=min(dev_size, len(df_vlsp_dev)),
        stratify=df_vlsp_dev["label"],
        random_state=seed
    )

    df_vihsd_dev_sampled = df_vihsd_dev_sampled.copy()
    df_vlsp_dev_sampled = df_vlsp_dev_sampled.copy()
    df_vihsd_dev_sampled["source"] = "vihsd"
    df_vlsp_dev_sampled["source"] = "vlsp"

    df_continual_dev = pd.concat([df_vlsp_dev_sampled, df_vihsd_dev_sampled], ignore_index=True)
    print(f"[PREPARE] Created joint validation set: total={len(df_continual_dev)} (ViHSD={len(df_vihsd_dev_sampled)}, VLSP={len(df_vlsp_dev_sampled)})")

    # 6. Save continual learning datasets
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_dev_path), exist_ok=True)

    df_continual_train.to_parquet(output_train_path, index=False)
    df_continual_dev.to_parquet(output_dev_path, index=False)
    print(f"[PREPARE] Saved training set to {output_train_path}")
    print(f"[PREPARE] Saved validation set to {output_dev_path}")

    # Calculate dynamic class weights for Focal Loss / CE Loss balance
    labels = df_continual_train["label"].values
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights / class_weights[0]  # normalize w.r.t CLEAN
    class_weights_dict = {
        "CLEAN": round(float(class_weights[0]), 4),
        "OFFENSIVE": round(float(class_weights[1]), 4),
        "HATE": round(float(class_weights[2]), 4)
    }
    print(f"[PREPARE] Dynamic Class Weights: {class_weights_dict}")
    return class_weights_dict

def main():
    parser = argparse.ArgumentParser(description="Prepare VLSP-2019 dataset for Continual Learning")
    parser.add_argument("--vlsp-dir", default="data/raw/vlsp-2019", help="Path to raw VLSP-2019 directory")
    parser.add_argument("--vihsd-train", default="data/processed/train.parquet", help="Path to ViHSD train parquet")
    parser.add_argument("--vihsd-dev", default="data/processed/dev.parquet", help="Path to ViHSD dev parquet")
    parser.add_argument("--output-train", default="data/processed/continual_train.parquet", help="Path to output train parquet")
    parser.add_argument("--output-dev", default="data/processed/continual_dev.parquet", help="Path to output dev parquet")
    parser.add_argument("--rehearsal-size", type=int, default=2500, help="Size of rehearsal buffer")
    parser.add_argument("--dev-size", type=int, default=500, help="Size of validation samples from each dataset")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    prepare_data(
        vlsp_dir=args.vlsp_dir,
        vihsd_train_path=args.vihsd_train,
        vihsd_dev_path=args.vihsd_dev,
        output_train_path=args.output_train,
        output_dev_path=args.output_dev,
        rehearsal_size=args.rehearsal_size,
        dev_size=args.dev_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
