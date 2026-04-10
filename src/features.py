import librosa
import numpy as np
import torch
from torch_geometric.data import Data


def extract_stress_features(y, sr=22050):
    """
    Extracts voice stress markers: Jitter, Shimmer, and Fundamental Frequency.
    Optimized for speed using librosa.yin instead of pyin.
    """
    try:
        # FFT-based Pitch tracking using yin (significantly faster than pyin)
        # Increased hop_length to 1024 as we only need global stats
        f0 = librosa.yin(
            y, fmin=60, fmax=500, sr=sr, frame_length=2048, hop_length=1024
        )
        
        # Simple voiced/unvoiced detection using amplitude threshold
        # Screams/Distress are usually high-intensity voiced sounds
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        voiced_mask = rms > (np.mean(rms) * 0.5)
        
        f0_voiced = f0[voiced_mask & (f0 > 0)]

        if len(f0_voiced) < 2:
            return 0.0, 0.0, 0.0, 0.0

        # F0 statistics
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)

        # Jitter: relative average variation of F0
        f0_diffs = np.abs(np.diff(f0_voiced))
        jitter = np.mean(f0_diffs) / f0_mean if f0_mean > 0 else 0.0

        # Shimmer: relative average variation of amplitude
        # Using peak amplitudes for speed
        amp_diffs = np.abs(np.diff(rms[voiced_mask]))
        shimmer = np.mean(amp_diffs) / np.mean(rms[voiced_mask]) if np.mean(rms[voiced_mask]) > 0 else 0.0

        return float(f0_mean), float(f0_std), float(jitter), float(shimmer)

    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def extract_features(audio_path, sr=22050, n_mfcc=20):
    """
    Extracts audio features from an audio file.
    Features: MFCC (20), Spectral Centroid, Zero Crossing Rate, RMS Energy, etc.
    Output shape: (n_frames, n_features)
    """
    try:
        # Load audio with a default duration of 2s for consistency
        y, sr = librosa.load(audio_path, sr=sr, duration=2.0)
    except Exception:
        return None

    return extract_features_from_array(y, sr=sr, n_mfcc=n_mfcc)


def extract_features_from_array(y, sr=22050, n_mfcc=20, include_stress=True):
    """
    Extracts features from a numpy array (audio buffer).
    Included: MFCCs, Spectral Centroid, ZCR, RMS, Bandwidth, Flatness + Jitter/Shimmer/F0.

    Args:
        y: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCCs
        include_stress: Whether to include stress features (Jitter, Shimmer, F0)

    Returns:
        features: Feature matrix (T, F) where F = 25 (+ 5 stress features if include_stress=True)
    """
    # Audio Normalization: Scale to [-1.0, 1.0] to handle volume differences
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    n_fft = 1024
    hop_length = 512

    # 1. MFCCs (20)
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )

    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    # 3. Best Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length
    )

    # 4. RMS Energy
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

    # 5. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )

    # 6. Spectral Flatness (Crucial for distinguishing noise from tonal screams)
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    )

    # Concatenate standard features
    features = np.vstack([mfccs, spectral_centroid, zcr, rms, bandwidth, flatness])
    features = features.T

    # Add stress features as global statistics (repeated for each frame)
    if include_stress:
        f0_mean, f0_std, jitter, shimmer = extract_stress_features(y, sr)

        # Create global stress features (repeated to match frame count)
        n_frames = features.shape[0]
        stress_feats = np.array(
            [[f0_mean, f0_std, jitter, shimmer] for _ in range(n_frames)]
        )

        # Concatenate
        features = np.hstack([features, stress_feats])

    # Safety Check: Replace NaNs/Infs with 0 to prevent CUDA errors
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def extract_global_features(y, sr=22050, n_mfcc=20, include_stress=True):
    """
    Extracts global (sentence-level) features for SVM.
    Returns mean + std of all features.

    Returns:
        feature_vector: (50,) or (58,) depending on include_stress
    """
    feat_matrix = extract_features_from_array(
        y, sr=sr, n_mfcc=n_mfcc, include_stress=include_stress
    )

    if feat_matrix is None:
        return None

    # Mean and Std across time
    m = np.mean(feat_matrix, axis=0)
    s = np.std(feat_matrix, axis=0)

    return np.concatenate([m, s])


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

    mfccs = features[:, :20]  # (T, 20)
    global_mfccs = np.mean(mfccs, axis=0)
    return global_mfccs
