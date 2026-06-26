import os
import sys
import json
import time
import shutil
import zipfile
import threading
from pathlib import Path
import pytest
import numpy as np

# Ensure root path is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import HateSpeechClassifier
from src.export.deploy_cl_model import deploy_model, write_active_version


class MockHateSpeechClassifier(HateSpeechClassifier):
    def __init__(self, artifact_dir="tmp_artifacts", use_word_segmentation=False):
        # Initialize variables manually to avoid downloading/loading heavy model files during tests.
        self.model_source = "local"
        self.hf_repo_id = None
        self.model_path = "tmp_model"
        self.artifact_dir = Path(artifact_dir)
        self.label_mapping_path = self.artifact_dir / "label_mapping.json"
        self.metadata_path = self.artifact_dir / "metadata.json"
        self.threshold = 0.5
        self.thresholds = None
        self.max_length = 128
        self.device_name = "cpu"
        self.use_word_segmentation = use_word_segmentation
        self.model = object()  # mock non-None model
        self.tokenizer = object()  # mock non-None tokenizer
        self.device = "cpu"
        self.label_mapping = {"id2label": {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}}
        self.id2label = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
        self.metadata = {"model_version": "v2.0.0-mock"}
        self.loaded_from = "local"
        self.temperature = None
        self.features_config = {"enable_toxicity_score": False, "enable_toxic_spans": False}
        
        self.lock = threading.Lock()
        self.is_reloading = False
        self.active_version_path = self.artifact_dir / "active_version.json"

    def _load(self):
        # Mock load that takes time to simulate a real file load
        time.sleep(0.5)
        self.model = object()
        self.tokenizer = object()
        self.device = "cpu"


def test_concurrency_hot_reloading_non_blocking():
    """Verify that when reload_model() is executing, predict() falls back immediately
    without waiting for the reload to complete (non-blocking).
    """
    classifier = MockHateSpeechClassifier()
    
    # 1. Start reload_model on a background thread.
    # It will sleep for 0.5 seconds inside _load().
    reload_thread = threading.Thread(target=classifier.reload_model, args=("new_mock_path",))
    
    # Measure latency of predict while reload is in progress
    start_time = time.time()
    reload_thread.start()
    
    # Wait a small fraction to make sure the thread has started and set is_reloading = True
    time.sleep(0.05)
    assert classifier.is_reloading is True
    
    # Call predict() - it should return fallback CLEAN immediately
    predict_start = time.time()
    res = classifier.predict("Some test sentence.")
    predict_duration = time.time() - predict_start
    
    # 2. Check the response is immediate (less than 50 milliseconds to account for test overhead)
    assert predict_duration < 0.050, f"predict() was blocked! Latency: {predict_duration:.4f}s"
    assert res["label"] == "CLEAN"
    assert res["confidence"] == 1.0
    
    # Wait for the reload thread to finish
    reload_thread.join()
    assert classifier.is_reloading is False


def test_dynamic_staging_and_fallback_on_locked_dir(tmp_path):
    """Test that deploy_model uses dynamic staging and falls back to a versioned path
    if the main destination directory is simulated to be locked/inaccessible.
    """
    dest_dir = tmp_path / "hate_speech_model"
    dest_dir.mkdir()
    
    # Create a dummy metadata and label mapping in dest_dir
    with open(dest_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}}, f)
    with open(dest_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"model_version": "v2.0.0"}, f)
        
    # Create a mock zip file with model config
    zip_path = tmp_path / "CL_output.zip"
    success_marker = tmp_path / "CL_output.zip.success"
    
    model_dir = tmp_path / "temp_model_files"
    model_dir.mkdir()
    # Write a dummy config.json
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"model_type": "xlm-roberta"}, f)
        
    # Create the zip archive
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        zip_file.write(model_dir / "config.json", "config.json")
        
    # Create success marker
    with open(success_marker, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "status": "SUCCESS"}, f)
        
    # Let's mock HateSpeechClassifier in the module during test to avoid loading dependencies.
    import src.export.deploy_cl_model as deploy_module
    
    class FakeClassifierForDeploy:
        def __init__(self, **kwargs):
            pass
        def predict(self, text):
            return {"label": "CLEAN", "confidence": 1.0}
            
    original_classifier = deploy_module.HateSpeechClassifier
    deploy_module.HateSpeechClassifier = FakeClassifierForDeploy
    
    try:
        # Mock main model directory to fail rename (simulating locked files)
        # We can do this by patching os.rename and os.replace
        original_rename = os.rename
        original_replace = os.replace
        
        def mock_rename(src, dst):
            # If destination is the target model dir, raise PermissionError
            if str(dst).endswith("model"):
                raise PermissionError("Access is denied (locked)")
            return original_rename(src, dst)
            
        def mock_replace(src, dst):
            if str(dst).endswith("model"):
                raise PermissionError("Access is denied (locked)")
            return original_replace(src, dst)
            
        # Hook mocks
        os.rename = mock_rename
        os.replace = mock_replace
        
        # Run deployment
        deployed = deploy_model(
            zip_path=str(zip_path),
            success_marker=str(success_marker),
            dest_dir=str(dest_dir),
            retry_count=1,
            retry_delay=0.1
        )
        
        # Deployment should succeed via fallback versioned directory!
        assert deployed is True
        
        # Verify active_version.json exists and points to a model_v<timestamp> directory
        active_version_path = dest_dir / "active_version.json"
        assert active_version_path.exists()
        with open(active_version_path, "r", encoding="utf-8") as f:
            active_data = json.load(f)
            assert "active_path" in active_data
            assert "model_v" in active_data["active_path"]
            
        # Check that the versioned folder exists
        fallback_path = Path(active_data["active_path"])
        assert fallback_path.exists() or (dest_dir / fallback_path.name).exists()
        
    finally:
        # Restore original functions
        os.rename = original_rename
        os.replace = original_replace
        deploy_module.HateSpeechClassifier = original_classifier
