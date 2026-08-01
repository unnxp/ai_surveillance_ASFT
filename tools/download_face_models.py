import os
import zipfile
import urllib.request

def download_and_extract():
    # Target files and directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    
    zip_path = os.path.join(models_dir, "buffalo_l.zip")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    # Check if files already exist
    det_model_path = os.path.join(models_dir, "det_10g.onnx")
    rec_model_path = os.path.join(models_dir, "w600k_r50.onnx")
    
    if os.path.exists(det_model_path) and os.path.exists(rec_model_path):
        print("Models already exist in models/ folder. Skipping download.")
        return
        
    try:
        if not os.path.exists(zip_path):
            print(f"Downloading face models from {url}...")
            print("This might take a few minutes as the file is ~280MB. Please wait...")
            
            # Progress hook helper
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                percent = min(100, percent)
                print(f"\rProgress: {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="")
                
            urllib.request.urlretrieve(url, zip_path, progress_hook)
            print("\nDownload complete.")
        else:
            print("buffalo_l.zip already exists. Skipping download.")
            
        print("Extracting files...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We only extract det_10g.onnx and w600k_r50.onnx
            for file_info in zip_ref.infolist():
                filename = os.path.basename(file_info.filename)
                if filename in ["det_10g.onnx", "w600k_r50.onnx"]:
                    # Extract to models_dir directly
                    target_path = os.path.join(models_dir, filename)
                    with zip_ref.open(file_info) as source:
                        with open(target_path, "wb") as target:
                            target.write(source.read())
                    print(f"Extracted: {filename} -> {target_path}")
                    
        # Remove downloaded zip file to clean up space
        os.remove(zip_path)
        print("Cleaned up temporary zip file.")
        print("Successfully installed SCRFD and ArcFace models!")
    except Exception as e:
        print(f"\nError: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    download_and_extract()
