import os
import sys
import numpy as np
import torch
import random
import json
import librosa

# Add the project root to sys.path to allow importing from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import features, models
from torch_geometric.loader import DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_CACHE = r"C:\Users\ASUS\.cache\kagglehub\datasets\whats2000\human-screaming-detection-dataset\versions\2"


def get_file_paths(dataset_path):
    scream, non_scream = [], []
    for root, dirs, files in os.walk(dataset_path):
        folder = os.path.basename(root).lower()
        for file in files:
            if file.endswith(".wav"):
                fpath = os.path.join(root, file)
                # Check folder name contains 'notscream' for non-scream
                if "notscreaming" in folder:
                    non_scream.append(fpath)
                # Check for 'screaming' in folder
                elif "screaming" == folder:
                    scream.append(fpath)
                # Fallback: check if 'notscream' anywhere in path
                elif "notscream" in root.lower():
                    non_scream.append(fpath)
                elif "scream" in root.lower():
                    scream.append(fpath)
    return scream, non_scream


print("=== Training High-Accuracy Multi-Task Model ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading dataset...")
scream_files, non_scream_files = get_file_paths(DATASET_CACHE)
print(f"Found: {len(scream_files)} screams, {len(non_scream_files)} non-screams")

# Balance dataset - use more samples
max_per_class = 800
scream_files = scream_files[:max_per_class]
non_scream_files = non_scream_files[:max_per_class]
all_files = [(f, 1) for f in scream_files] + [(f, 0) for f in non_scream_files]
random.shuffle(all_files)

print(f"\nProcessing {len(all_files)} files...")
data_list = []

# Apply data augmentation for minority class (screams)
for i, (fpath, label) in enumerate(all_files):
    try:
        y, sr = librosa.load(fpath, sr=22050, duration=2.0)
        if len(y) < 11025:
            continue
        feats = features.extract_features_from_array(y, sr=sr, include_stress=True)
        if feats is None:
            continue
        g = features.build_graph_from_features(feats)
        if g is None:
            continue
        g.y_scream = torch.tensor([label], dtype=torch.long)
        g.y_emotion = torch.tensor([0], dtype=torch.long)
        g.domain = torch.tensor([0], dtype=torch.long)
        data_list.append(g)

        # Augment scream samples with noise
        if label == 1 and i % 2 == 0:
            y_aug = y + np.random.randn(len(y)) * 0.005
            feats_aug = features.extract_features_from_array(
                y_aug, sr=sr, include_stress=True
            )
            if feats_aug is not None:
                g_aug = features.build_graph_from_features(feats_aug)
                if g_aug:
                    g_aug.y_scream = torch.tensor([label], dtype=torch.long)
                    g_aug.y_emotion = torch.tensor([0], dtype=torch.long)
                    g_aug.domain = torch.tensor([0], dtype=torch.long)
                    data_list.append(g_aug)
    except:
        pass
    if (i + 1) % 300 == 0:
        print(f"  {i + 1}/{len(all_files)}")

print(f"Processed: {len(data_list)} samples")
num_features = data_list[0].num_node_features

# Split
split = int(len(data_list) * 0.85)
train_data, val_data = data_list[:split], data_list[split:]
random.shuffle(train_data), random.shuffle(val_data)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Model
print("\nBuilding model...")
model = models.MultiHeadGGNN(
    num_node_features=num_features,
    hidden_channels=128,
    num_layers=6,
    num_emotion_classes=5,
    lstm_hidden=128,
    use_dann=True,
).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

best_acc = 0
best_state = None
patience = 0

print("\nTraining...")
for epoch in range(60):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        scream_logits, emotion_logits, domain_logits = model(batch)
        loss = (
            0.6 * torch.nn.functional.nll_loss(scream_logits, batch.y_scream.squeeze())
            + 0.3
            * torch.nn.functional.nll_loss(emotion_logits, batch.y_emotion.squeeze())
            + 0.1
            * torch.nn.functional.nll_loss(domain_logits, 1 - batch.domain.squeeze())
        )
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)[0].argmax(dim=1)
            correct += (pred == batch.y_scream.squeeze()).sum().item()

    acc = correct / len(val_loader.dataset)
    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1

    print(f"  Epoch {epoch + 1}/60 | Acc: {acc:.2%} | Best: {best_acc:.2%}")

    if patience >= 10:
        print("Early stopping!")
        break

# Save
if best_state:
    model.load_state_dict(best_state)

os.makedirs("scream_models", exist_ok=True)
torch.save(model.state_dict(), "scream_models/scream_ggnn.pt")

config = {
    "num_node_features": num_features,
    "hidden_channels": 128,
    "num_layers": 6,
    "num_emotion_classes": 5,
    "lstm_hidden": 128,
    "scream_accuracy": float(best_acc),
    "model_type": "MultiHeadGGNN",
    "use_dann": True,
    "emotions": ["Neutral", "Happy", "Sad", "Angry", "Fearful"],
    "last_trained": str(np.datetime64('now')),
}
with open("scream_models/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n=== DONE === Best Accuracy: {best_acc:.2%}")
print("Model saved to: scream_models/scream_ggnn.pt")
