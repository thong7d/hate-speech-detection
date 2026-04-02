# src/data/download.py
import os
import requests
import zipfile
import shutil
from omegaconf import OmegaConf

def download_and_extract(repo_path: str, drive_root: str):
    """
    Downloads the ViHSD zip file from GitHub and extracts the CSV files
    directly into the configured raw data directory on Google Drive.
    """
    # 1. Load configuration
    config_path = os.path.join(repo_path, "configs/paths.yaml")
    cfg = OmegaConf.load(config_path)
    
    # 2. Setup paths safely
    raw_dir = cfg.data.raw_dir.replace("{drive_root}", drive_root)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Ensure we use the 'raw' GitHub URL for binary download
    zip_url = "https://github.com/sonlam1102/vihsd/raw/main/data/vihsd.zip"
    zip_temp_path = "/tmp/vihsd_temp.zip"
    target_files = ["train.csv", "dev.csv", "test.csv"]
    
    print(f"Downloading dataset from: {zip_url}")
    
    # 3. Download the ZIP file
    response = requests.get(zip_url, stream=True)
    if response.status_code == 200:
        with open(zip_temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Download successful.")
    else:
        raise Exception(f"❌ Failed to download file. HTTP Status: {response.status_code}")

    # 4. Extract and Flatten (ignore internal folders)
    print("Extracting files...")
    extracted_count = 0
    with zipfile.ZipFile(zip_temp_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            filename = os.path.basename(member.filename)
            
            # Skip empty names (directories) or hidden OS files
            if not filename or filename.startswith('.'):
                continue
                
            if filename in target_files:
                source = zip_ref.open(member)
                target_path = os.path.join(raw_dir, filename)
                
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"✅ Extracted: {filename} -> {target_path}")
                extracted_count += 1

    # 5. Cleanup
    os.remove(zip_temp_path)
    
    if extracted_count == len(target_files):
        print("🎉 Phase 1: Data Acquisition completed successfully.")
    else:
        print(f"⚠️ Warning: Expected {len(target_files)} files, but extracted {extracted_count}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        download_and_extract(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python download.py <repo_path> <drive_root>")