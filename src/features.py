import librosa
import numpy as np
import torch
from torch_geometric.data import Data

def extract_features(audio_path, sr=22050, n_mfcc=20):
    """
    Extracts audio features from an audio file.
    Features: MFCC (20), Spectral Centroid, Zero Crossing Rate, RMS Energy.
    Output shape: (n_frames, n_features)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    return extract_features_from_array(y, sr=sr, n_mfcc=n_mfcc)
    
    # Concatenate features
    # (n_features, T) -> (T, n_features)
    features = np.vstack([mfccs, spectral_centroid, zcr, rms])
    features = features.T
    
    return features

def extract_features_from_array(y, sr=22050, n_mfcc=20):
    """
    Extracts features from a numpy array (audio buffer).
    Included: MFCCs, Spectral Centroid, ZCR, RMS, Bandwidth, and Flatness.
    """
    # Audio Normalization: Scale to [-1.0, 1.0] to handle volume differences
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    n_fft = 1024
    hop_length = 512
    
    # 1. MFCCs (20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 3. Best Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
    
    # 4. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

    # 5. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # 6. Spectral Flatness (Crucial for distinguishing noise from tonal screams)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    
    # Concatenate all features
    features = np.vstack([mfccs, spectral_centroid, zcr, rms, bandwidth, flatness])
    features = features.T
    
    # Safety Check: Replace NaNs/Infs with 0 to prevent CUDA errors
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features

def build_graph_from_features(features):
    """
    Converts feature matrix (T, F) into a PyG Data object.
    Nodes: T frames
    Edges: t -> t+1 (Temporal adjacency)
    """
    if features is None:
        return None
        
    num_nodes = features.shape[0]
    if num_nodes == 0:
        return None

    # Node Features
    x = torch.tensor(features, dtype=torch.float)
    
    # Edges: t -> t+1
    # source nodes: 0, 1, ..., T-2
    # target nodes: 1, 2, ..., T-1
    if num_nodes > 1:
        source_nodes = torch.arange(0, num_nodes - 1, dtype=torch.long)
        target_nodes = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    return data

def process_file_for_svm(audio_path, sr=22050):
    """
    Extracts global averaged MFCCs for SVM.
    """
    features = extract_features(audio_path, sr=sr)
    if features is None:
        return None
    
    # Global average across time (T, F) -> (F,)
    # But wait, user said "Use global-averaged MFCCs".
    # My features include spectral centroid etc.
    # I should strictly follow "Use global-averaged MFCCs" for SVM?
    # Or "Include... as node features to enrich the MFCC data".
    # The SVM part says "Use global-averaged MFCCs as input".
    # I'll use the mean of the first 20 columns (MFCCs).
    
    mfccs = features[:, :20] # (T, 20)
    global_mfccs = np.mean(mfccs, axis=0)
    return global_mfccs
