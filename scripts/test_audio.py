import librosa
from src import data_loader, features
import os

# Get dataset path
print("Loading dataset...")
path = data_loader.download_dataset()
scream_files, non_scream_files = data_loader.get_file_paths(path)
print(f"Files: scream={len(scream_files)}, ambient={len(non_scream_files)}")

# Try single file
if scream_files:
    fpath = scream_files[0]
    print(f"Processing: {fpath}")
    y, sr = librosa.load(fpath, sr=22050, duration=2.0)
    print(f"Loaded: {len(y)} samples at {sr}Hz")

    print("Extracting features (without stress)...")
    feats = features.extract_features_from_array(y, sr=sr, include_stress=False)
    print(f"Features shape: {feats.shape}")

    print("Extracting features (with stress)...")
    feats_stress = features.extract_features_from_array(y, sr=sr, include_stress=True)
    print(f"Features with stress: {feats_stress.shape}")

    print("Building graph...")
    g_data = features.build_graph_from_features(feats)
    print(f"Graph: {g_data.num_nodes} nodes, {g_data.num_node_features} features")

    print("\nSUCCESS!")
else:
    print("No files found!")
