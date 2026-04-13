"""
src/features.py — Audio feature extraction.

Feature layout per frame (with stress, 49 total):
  [0:20]   MFCCs
  [20:40]  Delta-MFCCs  (velocity of spectral change)
  [40:45]  Spectral: centroid, ZCR, RMS, bandwidth, flatness
  [45:49]  Stress globals tiled: F0 mean/std, jitter, shimmer

Without stress (45 features): first 45 columns only.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import torch
from torch_geometric.data import Data

# ── Constants (must match config.json after training) ─────────────────────────
N_MFCC       = 20
N_FFT        = 1024
HOP_LENGTH   = 512
N_BASE_FEATS = 45   # 20 MFCC + 20 delta-MFCC + 5 spectral
N_STRESS     = 4
N_FEATURES   = N_BASE_FEATS + N_STRESS  # 49 (with stress)


# ── Stress features ───────────────────────────────────────────────────────────

def extract_stress_features(y, sr):
    """
    Estimate vocal stress: F0 mean/std, jitter (F0 variation), shimmer (amplitude variation).
    Returns (4,) float32 array; all zeros on failure.
    """
    try:
        f0 = librosa.yin(y, fmin=60, fmax=500, sr=sr,
                         frame_length=2048, hop_length=1024)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=1024)[0]
        voiced = (rms > np.mean(rms) * 0.3)[:len(f0)]
        f0_v = f0[voiced & (f0 > 0)]

        if len(f0_v) < 2:
            return np.zeros(4, dtype=np.float32)

        f0_mean = float(np.mean(f0_v))
        f0_std  = float(np.std(f0_v))
        jitter  = float(np.mean(np.abs(np.diff(f0_v))) / (f0_mean + 1e-8))

        rms_v = rms[voiced[:len(rms)]]
        rms_v = rms_v if rms_v.size > 1 else np.array([1.0])
        shimmer = float(np.mean(np.abs(np.diff(rms_v))) / (np.mean(rms_v) + 1e-8))

        return np.array([f0_mean, f0_std, jitter, shimmer], dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)


# ── Per-frame features ────────────────────────────────────────────────────────

def extract_features_from_array(y, sr=22050, n_mfcc=N_MFCC, include_stress=True):
    """
    Extract per-frame feature matrix from a waveform array.

    Returns:
        np.ndarray shape (T, 45) without stress, (T, 49) with stress — float32.
        None on failure.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    try:
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                           n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_delta = librosa.feature.delta(mfcc)
        sc  = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
        sf  = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Align all arrays to the same frame count
        T = mfcc.shape[1]
        def _t(a): return a[:, :T]
        mfcc_delta, sc, zcr, rms, sb, sf = (
            _t(mfcc_delta), _t(sc), _t(zcr), _t(rms), _t(sb), _t(sf)
        )

        feats = np.vstack([mfcc, mfcc_delta, sc, zcr, rms, sb, sf]).T  # (T, 45)

        if include_stress:
            stress = extract_stress_features(y, sr)          # (4,)
            feats  = np.hstack([feats, np.tile(stress, (T, 1))])  # (T, 49)

        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    except Exception:
        return None


def extract_features(audio_path, sr=22050, n_mfcc=N_MFCC):
    """Load audio file (2 s) and return per-frame feature matrix."""
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=2.0)
        return extract_features_from_array(y, sr, n_mfcc)
    except Exception:
        return None


def extract_global_features(y, sr=22050):
    """
    Return mean+std global feature vector for SVM input.
    Shape: (90,) — 45 mean + 45 std (no stress features).
    """
    feats = extract_features_from_array(y, sr, include_stress=False)
    if feats is None:
        return None
    return np.concatenate([np.mean(feats, axis=0), np.std(feats, axis=0)])


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph_from_features(feats):
    """
    Convert a (T, F) feature matrix into a PyG Data graph.
    Nodes = temporal frames.  Edges = bidirectional t ↔ t+1.
    Returns None when feats is None or has fewer than 2 frames.
    """
    if feats is None or feats.shape[0] < 2:
        return None

    T = feats.shape[0]
    x = torch.tensor(feats, dtype=torch.float)

    src = list(range(T - 1))
    dst = list(range(1, T))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)
