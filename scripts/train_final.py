"""
scripts/train_final.py — Unified multi-task training script.

Trains MultiHeadGGNN jointly on:
  Head A  Binary scream detection     (scream vs ambient)
  Head B  5-class emotion type        (Neutral / Happy / Sad / Angry / Fearful)
  Head C  Domain adversarial (DANN)   (scream-dataset vs emotion-dataset)

Datasets
────────
  • Human Screaming Detection (Kaggle: whats2000/human-screaming-detection-dataset)
  • RAVDESS  (Kaggle: uwrfkaggler/ravdess-emotional-speech-audio)
  • CREMA-D  (Kaggle: ejlok1/cremad)
  • TESS     (Kaggle: ejlok1/toronto-emotional-speech-set-data)

Outputs  (all in scream_models/)
────────
  multitask_ggnn.pt   model weights
  scream_svm.pkl      SVM baseline weights
  config.json         architecture + accuracy metadata

Usage
─────
  cd aishee_capstone
  python scripts/train_final.py
"""

import os
import sys
import json
import random
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import joblib
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import features, models, data_loader
from src.data_loader import CANONICAL_MAP, EMOTION_LABELS

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    # Architecture
    "hidden":        256,
    "num_layers":    8,
    "lstm_hidden":   256,   # trunk width in new architecture

    # Training
    "lr":            0.0008,    # Slightly lower for finer convergence
    "weight_decay":  0.02,      # Double weight decay to prevent overfitting
    "batch_size":    64,
    "epochs":        150,       # Increased for 95% push
    "patience":      999,       # Effectively removed early stopping
    "clip_norm":     0.8,       # Tighter clip for stability
    "focal_gamma":   2.0,
    "label_smoothing": 0.1,    # Help prevent overconfidence

    # Data
    "sr":            22050,
    "clip_duration": 2.0,
    "max_scream":    900,       # binary detection samples per class
    "max_per_class": 2500,      # emotion samples per class (before aug)
    "augment_to":    2000,      # target per emotion class after augmentation

    # Paths
    "output_dir":    "scream_models",
}

NUM_CLASSES = len(EMOTION_LABELS)  # 5
SR          = CFG["sr"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print("  Multi-Task Scream + Emotion Training")
print("=" * 55)
print(f"  Device  : {DEVICE}")
print(f"  Classes : {EMOTION_LABELS}")


# ══════════════════════ AUGMENTATION ══════════════════════════════════════════

def augment(y, sr):
    """Apply a chain of random augmentations to combat overfitting."""
    try:
        # 1. White Noise injection (80% chance)
        if random.random() < 0.8:
            noise_amp = random.uniform(0.001, 0.006)
            y = y + noise_amp * np.random.normal(size=y.shape).astype(np.float32)

        # 2. Time Stretching (50% chance)
        if random.random() < 0.5:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.8, 1.2))

        # 3. Pitch Shifting (60% chance)
        if random.random() < 0.6:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-4, 4))

        # 4. Random Gain (90% chance)
        if random.random() < 0.9:
            y = y * random.uniform(0.5, 1.3)

    except Exception:
        pass
    return y.astype(np.float32)


# ══════════════════════ GRAPH BUILDING ════════════════════════════════════════

def _make_graph(y, sr, s_label, e_label, domain, do_augment=False):
    """Audio array → PyG Data graph.  Returns None on failure."""
    try:
        if do_augment:
            y = augment(y, sr)
        feats = features.extract_features_from_array(y, sr=sr, include_stress=True)
        g     = features.build_graph_from_features(feats)
        if g is None:
            return None
        g.y_scream  = torch.tensor([s_label], dtype=torch.long)
        g.y_emotion = torch.tensor([e_label], dtype=torch.long)
        g.domain    = torch.tensor([domain],  dtype=torch.long)
        return g
    except Exception:
        return None


def _load(fpath, sr=SR, duration=None):
    """Load a wav file, return float32 array or None."""
    try:
        y, _ = librosa.load(fpath, sr=sr, duration=duration or CFG["clip_duration"])
        if len(y) < int(0.3 * sr):
            return None
        return y.astype(np.float32)
    except Exception:
        return None


# ══════════════════════ FOCAL LOSS ════════════════════════════════════════════

def focal_loss(log_probs, targets, gamma=2.0, weight=None, smoothing=0.1):
    """
    Focal loss with integrated label smoothing for high-accuracy regularization.
    """
    num_classes = log_probs.size(1)
    with torch.no_grad():
        # Smoothed targets
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)

    probs = torch.exp(log_probs)
    pt = (probs * true_dist).sum(1) # probability of the correct class
    base_loss = -(true_dist * log_probs).sum(1)

    if weight is not None:
        batch_weights = weight[targets]
        base_loss = base_loss * batch_weights

    return ((1.0 - pt) ** gamma * base_loss).mean()


# ══════════════════════ DATA LOADING ══════════════════════════════════════════

print("\n[1/4] Loading datasets …")

# ── Scream detection dataset ──────────────────────────────────────────────────
scream_root = data_loader.download_dataset()
scream_files, ambient_files = data_loader.get_file_paths(scream_root)
random.shuffle(scream_files);  scream_files  = scream_files[:CFG["max_scream"]]
random.shuffle(ambient_files); ambient_files = ambient_files[:CFG["max_scream"]]
print(f"  Screams: {len(scream_files)},  Ambient: {len(ambient_files)}")

# ── Emotion datasets ──────────────────────────────────────────────────────────
emotion_data = data_loader.get_emotion_scream_files()

# Group by canonical class
by_class = {i: [] for i in range(NUM_CLASSES)}
skipped = 0
for fpath, info in emotion_data.items():
    idx = CANONICAL_MAP.get(info["emotion"].lower(), -1)
    if idx < 0:
        skipped += 1
        continue
    by_class[idx].append((fpath, info))

print(f"  Emotion distribution (raw, {skipped} unknown labels skipped):")
for i, label in enumerate(EMOTION_LABELS):
    print(f"    {label:10s}: {len(by_class[i]):5d}")


# ══════════════════════ BUILD GRAPH DATASET ═══════════════════════════════════

print("\n[2/4] Building graphs …")
data_list = []

# ── Scream / ambient (domain=0) ───────────────────────────────────────────────
fearful_idx = CANONICAL_MAP["fearful"]
neutral_idx = CANONICAL_MAP["neutral"]

for fpath in tqdm(scream_files, desc="  Screams  (domain 0)"):
    y = _load(fpath)
    if y is None:
        continue
    g = _make_graph(y, SR, s_label=1, e_label=fearful_idx, domain=0)
    if g:
        data_list.append(g)
    # One augmented copy per scream
    g2 = _make_graph(y, SR, s_label=1, e_label=fearful_idx, domain=0, do_augment=True)
    if g2:
        data_list.append(g2)

for fpath in tqdm(ambient_files, desc="  Ambient  (domain 0)"):
    y = _load(fpath)
    if y is None:
        continue
    # Ambient audio is genuinely neutral/non-emotional — label is correct
    g = _make_graph(y, SR, s_label=0, e_label=neutral_idx, domain=0)
    if g:
        data_list.append(g)

# ── Emotion samples (domain=1) ────────────────────────────────────────────────
for class_idx in range(NUM_CLASSES):
    items = by_class[class_idx][:]
    random.shuffle(items)
    items = items[:CFG["max_per_class"]]

    processed = []          # (graph, raw_y) for augmentation
    for fpath, info in tqdm(items, desc=f"  {EMOTION_LABELS[class_idx]:10s} (domain 1)", leave=False):
        y = _load(fpath)
        if y is None:
            continue
        s_label = 1 if info.get("is_crime_scream", False) else 0
        g = _make_graph(y, SR, s_label=s_label, e_label=class_idx, domain=1)
        if g:
            processed.append((g, y, s_label))

    # Augment underrepresented classes to augment_to target
    n_need = max(0, CFG["augment_to"] - len(processed))
    aug_added = 0
    if n_need > 0 and processed:
        for i in range(n_need):
            g_src, y_src, sl = processed[i % len(processed)]
            g_aug = _make_graph(y_src, SR, s_label=sl, e_label=class_idx,
                                domain=1, do_augment=True)
            if g_aug:
                data_list.append(g_aug)
                aug_added += 1

    for g, _, _ in processed:
        data_list.append(g)

    print(f"    {EMOTION_LABELS[class_idx]:10s}: {len(processed):4d} orig + {aug_added:4d} aug")

print(f"\n  Total graphs : {len(data_list)}")

# ── Split ─────────────────────────────────────────────────────────────────────
random.shuffle(data_list)
split      = int(len(data_list) * 0.85)
train_data = data_list[:split]
val_data   = data_list[split:]
print(f"  Train: {len(train_data)},  Val: {len(val_data)}")

NUM_NODE_FEATS = data_list[0].num_node_features
print(f"  Node feature dim: {NUM_NODE_FEATS}")

# ── Class weights for focal loss ──────────────────────────────────────────────
emo_counts = Counter(g.y_emotion.item() for g in train_data)
total_emo  = sum(emo_counts.values())
cls_weights = torch.tensor(
    [total_emo / (NUM_CLASSES * max(emo_counts.get(i, 1), 1))
     for i in range(NUM_CLASSES)],
    dtype=torch.float, device=DEVICE,
)
print(f"  Emotion class weights: {cls_weights.cpu().numpy().round(3)}")


# ══════════════════════ MODEL ══════════════════════════════════════════════════

print("\n[3/4] Building model …")
model = models.MultiHeadGGNN(
    num_node_features   = NUM_NODE_FEATS,
    hidden_channels     = CFG["hidden"],
    num_layers          = CFG["num_layers"],
    num_emotion_classes = NUM_CLASSES,
    lstm_hidden         = CFG["lstm_hidden"],
    use_dann            = True,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable parameters: {n_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
)
# Cosine annealing with warm restarts — recovers from local minima
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)

train_loader = DataLoader(train_data, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=0)
val_loader   = DataLoader(val_data,   batch_size=CFG["batch_size"], num_workers=0)


# ══════════════════════ TRAINING LOOP ═════════════════════════════════════════

print("\n[4/4] Training …")
best_combined = 0.0
best_state    = None
patience_ctr  = 0

for epoch in range(1, CFG["epochs"] + 1):

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        s_out, e_out, d_out = model(batch)

        scream_loss = F.nll_loss(s_out, batch.y_scream.squeeze())
        emotion_loss = focal_loss(e_out, batch.y_emotion.squeeze(),
                                  gamma=CFG["focal_gamma"], weight=cls_weights,
                                  smoothing=CFG["label_smoothing"])
        # DANN: train encoder to confuse the domain discriminator (flip labels)
        domain_loss = F.nll_loss(d_out, 1 - batch.domain.squeeze())

        loss = 0.5 * scream_loss + 0.4 * emotion_loss + 0.1 * domain_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_norm"])
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    # ── Validate ────────────────────────────────────────────────────────────
    model.eval()
    s_ok = e_ok = total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            s_out, e_out, _ = model(batch)
            s_ok  += (s_out.argmax(1) == batch.y_scream.squeeze()).sum().item()
            e_ok  += (e_out.argmax(1) == batch.y_emotion.squeeze()).sum().item()
            total += batch.y_scream.size(0)

    s_acc    = s_ok / total
    e_acc    = e_ok / total
    combined = (s_acc + e_acc) / 2
    lr_now   = optimizer.param_groups[0]["lr"]

    print(
        f"  Ep {epoch:03d} | loss {epoch_loss/len(train_loader):.4f}"
        f" | scream {s_acc:.2%} | emotion {e_acc:.2%}"
        f" | combined {combined:.2%} | lr {lr_now:.1e}"
    )

    if combined > best_combined:
        best_combined = combined
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
    else:
        patience_ctr += 1
        if patience_ctr >= CFG["patience"]:
            print(f"  Early stopping at epoch {epoch}.")
            break

# ── Load best checkpoint ───────────────────────────────────────────────────────
if best_state:
    model.load_state_dict(best_state)
model.eval()

# ── Final validation ──────────────────────────────────────────────────────────
s_ok = e_ok = total = 0
with torch.no_grad():
    for batch in DataLoader(val_data, batch_size=CFG["batch_size"], num_workers=0):
        batch = batch.to(DEVICE)
        s_out, e_out, _ = model(batch)
        s_ok  += (s_out.argmax(1) == batch.y_scream.squeeze()).sum().item()
        e_ok  += (e_out.argmax(1) == batch.y_emotion.squeeze()).sum().item()
        total += batch.y_scream.size(0)

final_s = s_ok / total
final_e = e_ok / total
print(f"\n  ── Final val: Scream {final_s:.2%}  |  Emotion {final_e:.2%} ──")


# ══════════════════════ SVM TRAINING ══════════════════════════════════════════

print("\n  Training SVM baseline (binary scream detection) …")
svm_X, svm_y = [], []
for fpath, lbl in ([(f, 1) for f in scream_files] + [(f, 0) for f in ambient_files]):
    y = _load(fpath)
    if y is None:
        continue
    gf = features.extract_global_features(y, SR)
    if gf is not None:
        svm_X.append(gf)
        svm_y.append(lbl)

svm_X = np.array(svm_X)
svm_y = np.array(svm_y)

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(C=10.0, kernel="rbf", gamma="scale", probability=True)),
])
svm.fit(svm_X, svm_y)
print(f"  SVM trained on {len(svm_X)} samples.")


# ══════════════════════ SAVE ══════════════════════════════════════════════════

out = CFG["output_dir"]
os.makedirs(out, exist_ok=True)

ggnn_path = os.path.join(out, "multitask_ggnn.pt")
svm_path  = os.path.join(out, "scream_svm.pkl")
cfg_path  = os.path.join(out, "config.json")

torch.save(model.state_dict(), ggnn_path)
joblib.dump(svm, svm_path)

config = {
    "num_node_features":    NUM_NODE_FEATS,
    "hidden_channels":      CFG["hidden"],
    "num_layers":           CFG["num_layers"],
    "lstm_hidden":          CFG["lstm_hidden"],
    "num_emotion_classes":  NUM_CLASSES,
    "model_type":           "MultiHeadGGNN",
    "use_dann":             True,
    "scream_accuracy":      round(final_s, 6),
    "emotion_accuracy":     round(final_e, 6),
    "emotions":             EMOTION_LABELS,
    "last_trained":         datetime.now().isoformat(timespec="seconds"),
    "datasets":             ["scream_detection", "RAVDESS", "CREMA-D", "TESS"],
    "features":             "20_MFCC + 20_delta_MFCC + 5_spectral + 4_stress",
}
with open(cfg_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\n  Saved: {ggnn_path}")
print(f"  Saved: {svm_path}")
print(f"  Saved: {cfg_path}")
print(f"\n{'='*55}")
print(f"  Done!  Scream: {final_s:.2%}  |  Emotion: {final_e:.2%}")
print(f"{'='*55}")
