from __future__ import annotations

import os
import sys
import argparse
import yaml
import json
import time
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

try:
    from src.data.prepare_vlsp_data import prepare_data
    from src.training.trainer import train_from_config
    from src.models.classifier import HateSpeechClassifier, compute_metrics
    from src.evaluation.calibrate import calibrate_temperature, save_calibration_results
except ImportError:
    # Fallback to local import if package structure is not configured
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.data.prepare_vlsp_data import prepare_data
    from src.training.trainer import train_from_config
    from src.models.classifier import HateSpeechClassifier, compute_metrics
    from src.evaluation.calibrate import calibrate_temperature, save_calibration_results


def evaluate_model_on_dataset(classifier: HateSpeechClassifier, dataset_path: str) -> dict[str, float]:
    df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)
    texts = df["text"].tolist()
    labels = df["label"].astype(int).values
    
    print(f"[GATEKEEPER] Running evaluation on {len(texts)} samples from {dataset_path}...")
    logits = classifier.get_raw_logits_batch(texts)
    
    # Apply temperature if present
    if getattr(classifier, "temperature", None) is not None:
        logits = logits / classifier.temperature
        
    metrics = compute_metrics((logits, labels))
    return metrics


def run_continual_learning(
    vlsp_dir: str,
    vihsd_train: str,
    vihsd_dev: str,
    vihsd_test: str,
    output_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    label_smoothing: float,
    zip_path: str | None,
    vlsp_part: str = "all",
) -> None:
    print("[CL] Starting Continual Learning Pipeline...")
    
    # Clean up previous training checkpoint residue to avoid contamination and ValueError on resume mismatch
    checkpoint_dir = Path(output_dir) / "checkpoint"
    if checkpoint_dir.exists():
        print(f"[CL] Clearing previous training residue at {checkpoint_dir}...")
        shutil.rmtree(checkpoint_dir)
            
    # 1. Prepare VLSP + Rehearsal data
    output_train_parquet = "data/processed/continual_train.parquet"
    output_dev_parquet = "data/processed/continual_dev.parquet"
    
    class_weights = prepare_data(
        vlsp_dir=vlsp_dir,
        vihsd_train_path=vihsd_train,
        vihsd_dev_path=vihsd_dev,
        output_train_path=output_train_parquet,
        output_dev_path=output_dev_parquet,
        rehearsal_size=4000,
        dev_size=500,
        seed=42,
        vlsp_part=vlsp_part
    )
    
    # 2. Generate temp CL configuration based on configs/train.yaml
    base_config_path = "configs/train.yaml"
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found: {base_config_path}")
        
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Override paths to use continual learning splits
    config["data"]["train_path"] = output_train_parquet
    config["data"]["valid_path"] = output_dev_parquet
    config["data"]["test_path"] = vihsd_test
    
    # In continual learning, we start from the existing fine-tuned baseline model weights
    local_base_model = os.path.join(output_dir, "model")
    if os.path.exists(local_base_model) and os.path.exists(os.path.join(local_base_model, "config.json")):
        config["model"]["base_model"] = local_base_model
        print(f"[CL] Starting continual learning from local base model: {local_base_model}")
    else:
        hf_base_model = config["export"].get("hf_repo_id", "thong7d/vihsd-xlmr-base-hate-speech")
        config["model"]["base_model"] = hf_base_model
        print(f"[CL] Starting continual learning from Hugging Face base model: {hf_base_model}")
        
    # Override hyperparameters
    config["training"]["learning_rate"] = float(lr)
    config["training"]["num_epochs"] = int(epochs)
    config["training"]["batch_size"] = int(batch_size)
    config["training"]["label_smoothing"] = 0.1
    config["training"]["output_dir"] = os.path.join(output_dir, "checkpoint")
    config["export"]["artifact_dir"] = output_dir
    config["export"]["final_model_dir"] = os.path.join(output_dir, "model")
    config["evaluation"]["metrics_output_path"] = os.path.join(output_dir, "metrics.json")
    
    # In continual learning, we fall back to standard Cross-Entropy for label smoothing stability
    config["training"]["loss"] = "cross_entropy"
    # Enable moderate class weighting to prevent forgetting of minority classes without destroying precision
    config["training"]["class_weighting"] = "sqrt_balanced"
    # Freeze roberta.encoder during the first epoch to protect pre-trained features
    config["training"]["freeze_backbone_first_epoch"] = True
    
    # Disable augmentations that are intended only for initial baseline training
    for aug in ["robustness_augmentation", "contrastive_augmentation", "diacritic_augmentation", "class_oversampling"]:
        if aug in config["training"]:
            config["training"][aug]["enabled"] = False
    if "eda_augmentation" in config["training"]:
        config["training"]["eda_augmentation"]["enabled"] = False
        
    # Write temp yaml config
    temp_config_path = "configs/train_cl_temp.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    print(f"[CL] Generated temp configuration at {temp_config_path}")
    
    # 3. Train the model using the config
    train_result = train_from_config(config)
    print("[CL] Training completed successfully.")
    
    # 4. Gatekeeper check
    print("[CL] Starting Gatekeeper Check...")
    
    # Load thresholds from api config or use default optimal thresholds
    api_config_path = "configs/api.yaml"
    thresholds = {"CLEAN": 0.5, "OFFENSIVE": 0.38, "HATE": 0.32}
    if os.path.exists(api_config_path):
        try:
            with open(api_config_path, "r", encoding="utf-8") as f:
                api_cfg = yaml.safe_load(f)
                if "model" in api_cfg and "thresholds" in api_cfg["model"]:
                    thresholds = api_cfg["model"]["thresholds"]
                    print(f"[GATEKEEPER] Loaded thresholds from api config: {thresholds}")
        except Exception as e:
            print(f"[GATEKEEPER] Warning reading api thresholds: {e}")

    # Load newly trained model (with thresholds)
    new_classifier = HateSpeechClassifier(
        model_source="local",
        model_path=os.path.join(output_dir, "model"),
        artifact_dir=output_dir,
        device="auto",
        use_word_segmentation=False,
        thresholds=thresholds
    )
    
    # Evaluate new model on original ViHSD test set
    eval_metrics = evaluate_model_on_dataset(new_classifier, vihsd_test)
    new_macro_f1 = eval_metrics.get("macro_f1", 0.0)
    print(f"[GATEKEEPER] New model Macro F1 on original ViHSD test set: {new_macro_f1:.4f}")
    
    # Load baseline metrics to compare
    baseline_f1 = 0.6461  # Fallback baseline
    baseline_report_path = Path("results/finetune_report.json")
    if baseline_report_path.exists():
        try:
            with open(baseline_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                baseline_f1 = report.get("macro_f1", 0.6461)
        except Exception as e:
            print(f"[GATEKEEPER] Warning reading baseline report: {e}")
            
    print(f"[GATEKEEPER] Baseline model Macro F1: {baseline_f1:.4f}")
    
    # Tolerable threshold: degradation should not exceed 1% (i.e. -0.01)
    threshold = baseline_f1 - 0.01
    if new_macro_f1 < threshold:
        warning_msg = f"[GATEKEEPER WARNING] New model F1 ({new_macro_f1:.4f}) is below baseline F1 ({baseline_f1:.4f}) and tolerable threshold ({threshold:.4f}). Catastrophic forgetting detected!"
        print(f"⚠️ {warning_msg}")
        print("[GATEKEEPER] Non-fatal mode: Proceeding to final steps to record results in reports...")
    else:
        print("✅ [GATEKEEPER PASSED] Performance on old data is preserved.")
    
    # 5. Temperature Recalibration on joint validation set
    print("[CL] Running temperature calibration on joint dev set...")
    dev_df = pd.read_parquet(output_dev_parquet)
    dev_texts = dev_df["text"].tolist()
    dev_labels = dev_df["label"].astype(int).values
    
    # Extract raw logits (uncalibrated)
    print("[CL] Extracting logits on joint validation set...")
    all_logits = []
    bs = 32
    for i in range(0, len(dev_texts), bs):
        batch = dev_texts[i:i + bs]
        batch_logits = new_classifier.get_raw_logits_batch(batch)
        all_logits.append(batch_logits)
    logits = np.vstack(all_logits)
    
    # Run calibration optimization
    calib_results = calibrate_temperature(
        logits=logits,
        labels=dev_labels,
        l2_lambda=0.01,
        init_temperature=1.5
    )
    
    print(f"[CL] Optimal Temperature found: {calib_results['optimal_temperature']}")
    print(f"[CL] ECE Before: {calib_results['ece_before']:.6f} -> ECE After: {calib_results['ece_after']:.6f}")
    
    # Save calibration results back to new metadata.json
    metadata_path = Path(output_dir) / "metadata.json"
    save_calibration_results(calib_results, str(metadata_path))
    print(f"[CL] Saved calibrated temperature to {metadata_path}")
    
    # Clean up temp configuration file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        
    # 6. Packaging model version zip if zip_path is specified
    if zip_path:
        print(f"[CL] Zipping model artifacts (excluding checkpoints) to {zip_path}...")
        zip_dir = Path(zip_path).parent
        if zip_dir:
            zip_dir.mkdir(parents=True, exist_ok=True)
            
        # Zip files under output_dir, excluding the massive checkpoint directories
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(output_dir):
                # Skip Trainer checkpoints which take up gigabytes of space
                if "checkpoint" in Path(root).parts:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, arcname)

        # Create a lightweight reports-only ZIP file (contains only json, txt, csv, md)
        reports_zip_path = zip_path.replace(".zip", "_reports.zip")
        print(f"[CL] Zipping lightweight reports to {reports_zip_path}...")
        with zipfile.ZipFile(reports_zip_path, 'w', zipfile.ZIP_DEFLATED) as r_zip:
            for root, dirs, files in os.walk(output_dir):
                # Skip checkpoints and weights
                if "checkpoint" in Path(root).parts or "model" in Path(root).parts:
                    continue
                for file in files:
                    if file.endswith((".json", ".txt", ".md", ".csv")):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        r_zip.write(file_path, arcname)
                    
        # Write .success file atomically AFTER zip is completely written
        success_marker_path = f"{zip_path}.success"
        with open(success_marker_path, "w") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "status": "SUCCESS",
                "macro_f1": new_macro_f1,
                "temperature": calib_results['optimal_temperature']
            }))
        print(f"[CL] Successfully created ZIP archive and atomic marker file {success_marker_path}")
        
    print("[CL] Continual Learning Pipeline finished successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual Learning Orchestrator for Hate Speech Classifier")
    parser.add_argument("--vlsp-dir", default="data/raw/vlsp-2019", help="Path to raw VLSP-2019 dataset directory")
    parser.add_argument("--vihsd-train", default="data/processed/train.parquet", help="Path to original ViHSD train parquet")
    parser.add_argument("--vihsd-dev", default="data/processed/dev.parquet", help="Path to original ViHSD dev parquet")
    parser.add_argument("--vihsd-test", default="data/processed/test.parquet", help="Path to original ViHSD test parquet")
    parser.add_argument("--output-dir", default="artifacts/hate_speech_model", help="Directory to save CL model version")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for incremental fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for CL training")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--label-smoothing", type=float, default=0.15, help="Label smoothing strength")
    parser.add_argument("--zip-path", default="artifacts/CL_output.zip", help="Path to output zip file (set None to skip zipping)")
    parser.add_argument("--vlsp-part", default="all", choices=["all", "1", "2"], help="VLSP split part for sequential CL")
    args = parser.parse_args()
    
    run_continual_learning(
        vlsp_dir=args.vlsp_dir,
        vihsd_train=args.vihsd_train,
        vihsd_dev=args.vihsd_dev,
        vihsd_test=args.vihsd_test,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        label_smoothing=args.label_smoothing,
        zip_path=args.zip_path if args.zip_path.lower() != "none" else None,
        vlsp_part=args.vlsp_part
    )


if __name__ == "__main__":
    main()
