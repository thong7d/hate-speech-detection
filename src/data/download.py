# src/data/download.py
import os
import requests
import zipfile
import shutil
import tempfile
from omegaconf import OmegaConf

def download_and_extract(config_path: str):
    """
    Downloads the ViHSD zip file from GitHub and extracts the CSV files
    directly into the configured raw data directory.
    """
    # 1. Load configuration
    cfg = OmegaConf.load(config_path)
    
    def resolve_path(template_path) -> str:
        """Resolve {project_root} placeholder."""
        if isinstance(template_path, str):
            # Here project_root is essentially the parent dir of configs/paths.yaml
            proj_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
            return template_path.replace("{project_root}", proj_root)
        if hasattr(template_path, "values"):
            for v in template_path.values():
                return resolve_path(v)
        return str(template_path)
        
    # 2. Setup paths safely
    raw_dir = resolve_path(cfg.data.raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Ensure we use the 'raw' GitHub URL for binary download
    zip_url = "https://github.com/sonlam1102/vihsd/raw/main/data/vihsd.zip"
    target_files = ["train.csv", "dev.csv", "test.csv"]
    
    print(f"Downloading dataset from: {zip_url}")
    
    # Tạo tệp tạm thời tương thích đa nền tảng (Windows/Linux)
    fd, zip_temp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd) # Giải phóng file handle cấp thấp ngay lập tức
    
    extracted_count = 0
    
    try:
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
        with zipfile.ZipFile(zip_temp_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                filename = os.path.basename(member.filename)
                
                # Skip empty names (directories) or hidden OS files
                if not filename or filename.startswith('.'):
                    continue
                    
                if filename in target_files:
                    target_path = os.path.join(raw_dir, filename)
                    
                    # Bổ sung khối 'with' cho biến source để hệ thống tự động giải phóng khóa tệp sau khi sao chép
                    with zip_ref.open(member) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
                    print(f"✅ Extracted: {filename} -> {target_path}")
                    extracted_count += 1

    finally:
        # 5. Cleanup (Luôn được thực thi, tránh lỗi WinError 32)
        if os.path.exists(zip_temp_path):
            os.remove(zip_temp_path)
    
    if extracted_count == len(target_files):
        print("🎉 Phase 1: Data Acquisition completed successfully.")
    else:
        print(f"⚠️ Warning: Expected {len(target_files)} files, but extracted {extracted_count}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        download_and_extract(sys.argv[1])
    else:
        print("Usage: python download.py <config_path>")