import os
import numpy as np
import torch
import random
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
# Add project root to sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import data_loader, features, models
import librosa
import json

# CACHED dataset path - only download if not exists
DATASET_CACHE = r"C:\Users\ASUS\.cache\kagglehub\datasets\whats2000\human-screaming-detection-dataset\versions\2"


def get_file_paths_cached(dataset_path):
    """Get file paths from cached dataset."""
    scream_files = []
    non_scream_files = []

    for root, dirs, files in os.walk(dataset_path):
        root_lower = root.lower()
        folder_name = os.path.basename(root).lower()
        for file in files:
            if file.endswith(".wav"):
                full_path = os.path.join(root, file)
                # Check folder name
                if "notscream" in folder_name:
                    non_scream_files.append(full_path)
                elif "scream" in folder_name:
                    scream_files.append(full_path)

    return scream_files, non_scream_files


# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("=== High-Accuracy Multi-Task Training ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Use cached dataset
if os.path.exists(DATASET_CACHE):
    print(f"Using cached dataset: {DATASET_CACHE}")
    scream_files, non_scream_files = get_file_paths_cached(DATASET_CACHE)
else:
    print("Dataset not in cache, downloading...")
    path = data_loader.download_dataset()
    scream_files, non_scream_files = data_loader.get_file_paths(path)

print(f"Full dataset: {len(scream_files)} scream + {len(non_scream_files)} ambient")


# Data augmentation functions
def augment_audio(y, sr):
    """Apply data augmentation to audio."""
    aug_type = random.choice(["none", "noise", "stretch", "pitch"])

    if aug_type == "noise":
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise
    elif aug_type == "stretch":
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)
    elif aug_type == "pitch":
        steps = random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    return y


def process_file(fpath, label, augment=False):
    """Process audio file with optional augmentation."""
    try:
        y, sr = librosa.load(fpath, sr=22050, duration=2.0)
        if len(y) < 11025:
            return None

        if augment:
            y = augment_audio(y, sr)

        feats = features.extract_features_from_array(y, sr=sr, include_stress=True)
        if feats is None:
            return None

        g_data = features.build_graph_from_features(feats)
        if g_data is None:
            return None

        g_data.y_scream = torch.tensor([label], dtype=torch.long)
        g_data.y_emotion = torch.tensor([0], dtype=torch.long)
        g_data.domain = torch.tensor([0], dtype=torch.long)

        return g_data
    except:
        return None


# Process all data (balanced)
print("\nProcessing audio files...")
data_list = []

max_per_class = 800  # Use more samples
scream_files_used = scream_files[:max_per_class]
ambient_files_used = non_scream_files[:max_per_class]

all_files = [(f, 1) for f in scream_files_used] + [(f, 0) for f in ambient_files_used]
random.shuffle(all_files)

for i, (fpath, label) in enumerate(all_files):
    g_data = process_file(fpath, label, augment=(label == 1))  # Augment screams
    if g_data:
        data_list.append(g_data)

    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(all_files)}...")

print(f"Total samples: {len(data_list)}")

if len(data_list) == 0:
    print("No data!")
    exit()

num_features = data_list[0].num_node_features
print(f"Feature dimension: {num_features}")

# Split with stratification
print("\nSplitting data...")
scream_data = [d for d in data_list if d.y_scream.item() == 1]
ambient_data = [d for d in data_list if d.y_scream.item() == 0]

random.shuffle(scream_data)
random.shuffle(ambient_data)

split = int(0.85 * len(scream_data))
train_data = scream_data[:split] + ambient_data[:split]
val_data = scream_data[split:] + ambient_data[split:]

random.shuffle(train_data)
random.shuffle(val_data)

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Build model
print("\nBuilding model...")
model = models.MultiHeadGGNN(
    num_node_features=num_features,
    hidden_channels=128,  # Larger
    num_layers=6,  # Deeper
    num_emotion_classes=5,
    lstm_hidden=128,
    use_dann=True,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training setup
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)
criterion = torch.nn.CrossEntropyLoss()

# Class weights for imbalance
class_weights = torch.tensor([1.0, 1.0]).to(device)

best_acc = 0
best_state = None
patience_counter = 0

print("\nTraining...")
for epoch in range(80):
    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scream_logits, emotion_logits, domain_logits = model(batch)

        scream_loss = criterion(scream_logits, batch.y_scream.squeeze())
        emotion_loss = criterion(emotion_logits, batch.y_emotion.squeeze())
        domain_labels = 1 - batch.domain.squeeze()
        domain_loss = criterion(domain_logits, domain_labels)

        # Multi-task loss
        total_loss = 0.6 * scream_loss + 0.3 * emotion_loss + 0.1 * domain_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += scream_loss.item()

    # Validate
    model.eval()
    correct = 0
    total = 0
    scream_correct = 0
    scream_total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            scream_logits, _, _ = model(batch)
            pred = scream_logits.argmax(dim=1)

            correct += (pred == batch.y_scream.squeeze()).sum().item()
            total += batch.y_scream.size(0)

            # Track scream detection separately
            scream_mask = batch.y_scream == 1
            if scream_mask.sum() > 0:
                scream_correct += (
                    ((pred == batch.y_scream.squeeze()) & scream_mask).sum().item()
                )
                scream_total += scream_mask.sum().item()

    acc = correct / total
    scream_acc = scream_correct / scream_total if scream_total > 0 else 0

    # Update scheduler
    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        best_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"  Epoch {epoch + 1}/80 | Acc: {acc:.2%} | Scream Det: {scream_acc:.2%} | Best: {best_acc:.2%}"
    )

    # Early stopping
    if patience_counter >= 15:
        print("Early stopping!")
        break

# Load best model
if best_state:
    model.load_state_dict(best_state)

# Save
print("\nSaving model...")
os.makedirs("scream_models", exist_ok=True)
torch.save(model.state_dict(), "scream_models/multitask_ggnn.pt")

config = {
    "num_node_features": num_features,
    "hidden_channels": 128,
    "num_layers": 6,
    "scream_accuracy": float(best_acc),
}
with open("scream_models/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n=== COMPLETE ===")
print(f"Best Accuracy: {best_acc:.2%}")
print(f"Model saved to: scream_models/multitask_ggnn.pt")
