from __future__ import annotations

import os
import sys
import argparse
import json
import time
import shutil
import zipfile
from pathlib import Path

try:
    from src.models.classifier import HateSpeechClassifier
    from src.utils.config import resolve_path
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.classifier import HateSpeechClassifier
    from src.utils.config import resolve_path


def write_active_version(active_version_path: Path, active_path: str) -> None:
    data = {
        "active_path": str(active_path).replace("\\", "/"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(active_version_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[DEPLOY] Updated active_version.json pointing to: {active_path}")


def deploy_model(
    zip_path: str,
    success_marker: str,
    dest_dir: str,
    retry_count: int,
    retry_delay: float
) -> bool:
    zip_path_obj = Path(zip_path)
    success_marker_obj = Path(success_marker)
    dest_dir_obj = Path(dest_dir)
    
    if not success_marker_obj.exists():
        print(f"[DEPLOY] Success marker not found at {success_marker_obj}. Deployment skipped.")
        return False
        
    print(f"[DEPLOY] Detected success marker. Loading metadata...")
    try:
        with open(success_marker_obj, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            print(f"[DEPLOY] Marker Metadata: {metadata}")
    except Exception as e:
        print(f"[DEPLOY] Warning: failed to parse success marker metadata: {e}")
        
    if not zip_path_obj.exists():
        print(f"[DEPLOY] Error: success marker exists but zip archive not found at {zip_path_obj}!")
        return False
        
    # Generate timestamp and directory paths
    timestamp = int(time.time())
    staging_dir = dest_dir_obj / f"model_staging_{timestamp}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DEPLOY] Extracting model zip to dynamic staging directory: {staging_dir}")
    try:
        with zipfile.ZipFile(zip_path_obj, 'r') as zip_ref:
            zip_ref.extractall(staging_dir)
    except Exception as e:
        print(f"[DEPLOY] Error: failed to unzip model archive: {e}")
        # Clean up staging dir
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        return False
        
    # Check integrity of the model inside staging
    config_json = staging_dir / "config.json"
    if not config_json.exists():
        print(f"[DEPLOY] Error: invalid model structure, config.json missing in extracted files.")
        shutil.rmtree(staging_dir)
        return False
        
    print(f"[DEPLOY] Extract successful. Running load integrity test...")
    try:
        # Load model on CPU to verify it initializes and runs successfully
        test_classifier = HateSpeechClassifier(
            model_source="local",
            model_path=str(staging_dir),
            artifact_dir=str(dest_dir_obj),
            device="cpu",
            use_word_segmentation=False
        )
        test_res = test_classifier.predict("Kiểm tra khả năng tải mô hình.")
        print(f"[DEPLOY] Dry run successful. Prediction: {test_res['label']} ({test_res['confidence']:.4f})")
    except Exception as e:
        print(f"[DEPLOY] Error: load test integrity check failed: {e}")
        shutil.rmtree(staging_dir)
        return False
        
    # Clean up the test classifier variables to prevent locking the staging directory files
    del test_classifier
    import gc
    gc.collect()
    
    target_dir = dest_dir_obj / "model"
    success = False
    
    # Try atomic folder rename / replacement
    print(f"[DEPLOY] Attempting replacement of main model directory...")
    for attempt in range(retry_count):
        try:
            if target_dir.exists():
                # On Windows, rename/replace can fail if target_dir is locked or has active handles.
                # Try os.replace or rmtree + rename.
                try:
                    shutil.rmtree(target_dir)
                    os.rename(staging_dir, target_dir)
                except Exception:
                    os.replace(staging_dir, target_dir)
            else:
                os.rename(staging_dir, target_dir)
                
            write_active_version(dest_dir_obj / "active_version.json", str(target_dir))
            print(f"[DEPLOY] Model deployed successfully to main path.")
            success = True
            break
        except Exception as e:
            print(f"[DEPLOY] Attempt {attempt + 1} to replace main model directory failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            
    if not success:
        # Fallback mechanism: Keep it in a versioned folder and update active_version.json pointer
        print(f"[DEPLOY] Fallback: main model path is locked. Deploying to versioned path...")
        versioned_dir = dest_dir_obj / f"model_v{timestamp}"
        try:
            os.rename(staging_dir, versioned_dir)
            write_active_version(dest_dir_obj / "active_version.json", str(versioned_dir))
            print(f"[DEPLOY] Model deployed successfully to versioned path: {versioned_dir}")
            success = True
        except Exception as e:
            print(f"[DEPLOY] CRITICAL: Fallback deployment failed: {e}")
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            return False
            
    # Clean up zip archive and success marker to prevent repeat deployments
    try:
        os.remove(zip_path_obj)
        os.remove(success_marker_obj)
        print(f"[DEPLOY] Cleaned up deployment zip and success marker.")
    except Exception as e:
        print(f"[DEPLOY] Warning: failed to clean up zip or success marker: {e}")
        
    print(f"[DEPLOY] Deployment cycle finished successfully!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomic Deployment script for Continual Learning weights")
    parser.add_argument("--zip-path", default="artifacts/CL_output.zip", help="Path to CL output zip archive")
    parser.add_argument("--success-marker", default="artifacts/CL_output.zip.success", help="Path to success marker file")
    parser.add_argument("--dest-dir", default="artifacts/hate_speech_model", help="Directory where model artifacts are served")
    parser.add_argument("--retry-count", type=int, default=3, help="Max retry attempts for directory replacements")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Delay in seconds between retry attempts")
    args = parser.parse_args()
    
    deploy_model(
        zip_path=args.zip_path,
        success_marker=args.success_marker,
        dest_dir=args.dest_dir,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay
    )


if __name__ == "__main__":
    main()
