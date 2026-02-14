import os
import kagglehub
import pandas as pd
import numpy as np

def download_dataset():
    """
    Downloads the Human Screaming Detection Dataset using kagglehub.
    Returns the path to the dataset.
    """
    # Check if download already exists in 'data_download'
    target_dir = os.path.join(os.getcwd(), 'data_download')
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Rough check if populated
        if len(os.listdir(target_dir)) > 0:
            print(f"Dataset found in {target_dir}. Skipping download.")
            return target_dir

    try:
        # Check for kaggle.json in current directory and set env vars if present
        local_kaggle = os.path.join(os.getcwd(), 'kaggle.json')
        if os.path.exists(local_kaggle):
            print(f"Found kaggle.json at {local_kaggle}...")
            try:
                with open(local_kaggle, 'r') as f:
                    content = f.read().strip()
                
                import json
                try:
                    # Attempt to parse as JSON
                    creds = json.loads(content)
                    if isinstance(creds, dict) and 'username' in creds and 'key' in creds:
                        print("Setting KAGGLE_USERNAME and KAGGLE_KEY from JSON.")
                        os.environ['KAGGLE_USERNAME'] = creds.get('username', '')
                        os.environ['KAGGLE_KEY'] = creds.get('key', '')
                    else:
                        print("JSON found but missing 'username' or 'key'.")
                except json.JSONDecodeError:
                    # Not JSON, treat as raw token
                    if content.startswith('KGAT_') or len(content) > 10:
                        print("Treating kaggle.json as a raw KAGGLE_API_TOKEN.")
                        os.environ['KAGGLE_API_TOKEN'] = content
                    else:
                        print("kaggle.json content is neither valid JSON nor a recognized token.")
            except Exception as e:
                print(f"Error reading kaggle.json: {e}")

        # Assuming the dataset slug is 'whats2000/human-screaming-detection-dataset'
        path = kagglehub.dataset_download("whats2000/human-screaming-detection-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset with kagglehub: {e}")
        print("Attempting fallback with kaggle CLI...")

        try:
            # Fallback: Use Kaggle CLI directly
            import subprocess
            target_dir = os.path.join(os.getcwd(), 'data_download')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            cmd = ["kaggle", "datasets", "download", "-d", "whats2000/human-screaming-detection-dataset", "--unzip", "-p", target_dir]
            subprocess.run(cmd, check=True)
            print(f"Dataset downloaded via CLI to: {target_dir}")
            return target_dir
        except Exception as cli_error:
            print(f"CLI Fallback failed: {cli_error}")
            print("NOTE: 403 Forbidden usually means missing Kaggle API credentials.")
            print("Please ensure 'kaggle' is in your PATH and configured, or place 'kaggle.json' in the project root.")
            return None

def get_file_paths(dataset_path):
    """
    Crawls the dataset directory and returns lists of (path, label).
    Label 1 = Scream, 0 = Non-Scream.
    """
    if dataset_path is None:
        print("Dataset path is None. Skipping file crawling.")
        return [], []

    scream_files = []
    non_scream_files = []
    
    # Structure might be: /dataset/Scream/*.wav or similar.
    # Or /dataset/Positive/..., /dataset/Negative/...
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith('.wav'):
                full_path = os.path.join(root, file)
                root_lower = root.lower()
                # Heuristic for labeling based on folder names
                # We check for the negative case first to avoid 'screaming' matching 'notscreaming'
                if 'notscreaming' in root_lower or 'noise' in root_lower or 'negative' in root_lower:
                    non_scream_files.append(full_path)
                elif 'screaming' in root_lower or 'positive' in root_lower or 'scream' in root_lower:
                    scream_files.append(full_path)
    
    print(f"Found {len(scream_files)} scream files and {len(non_scream_files)} non-scream files.")
    return scream_files, non_scream_files

if __name__ == "__main__":
    download_dataset()
