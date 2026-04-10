import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
# Add project root to sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import data_loader, features, models

print("=== Quick Multi-Task Training ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load files
path = data_loader.download_dataset()
scream_files, non_scream_files = data_loader.get_file_paths(path)

# Use only first 200 of each
max_samples = 500
scream_files = scream_files[:max_samples]
non_scream_files = non_scream_files[:max_samples]
print(f"Using {len(scream_files)} scream + {len(non_scream_files)} ambient")

# Process files
print("\nProcessing audio...")
data_list = []

for i, fpath in enumerate(scream_files + non_scream_files):
    label = 1 if i < len(scream_files) else 0
    try:
        y, sr = 22050, 22050
        import librosa

        y, sr = librosa.load(fpath, sr=22050, duration=2.0)
        if len(y) < 22050 * 0.5:
            continue

        feats = features.extract_features_from_array(y, sr=sr, include_stress=True)
        if feats is None:
            continue

        g_data = features.build_graph_from_features(feats)
        if g_data is None:
            continue

        g_data.y_scream = torch.tensor([label], dtype=torch.long)
        g_data.y_emotion = torch.tensor([0], dtype=torch.long)
        g_data.domain = torch.tensor([0], dtype=torch.long)

        data_list.append(g_data)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}...")
    except Exception as e:
        pass

print(f"Total: {len(data_list)} samples")

if len(data_list) == 0:
    print("No data!")
    exit()

num_features = data_list[0].num_node_features
print(f"Features: {num_features}")

# Split
np.random.shuffle(data_list)
split = int(len(data_list) * 0.8)
train_data = data_list[:split]
val_data = data_list[split:]
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Model
print("\nBuilding model...")
model = models.MultiHeadGGNN(
    num_node_features=num_features,
    hidden_channels=64,
    num_layers=4,
    num_emotion_classes=5,
    lstm_hidden=64,
    use_dann=True,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train
print("\nTraining...")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0
for epoch in range(40):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        scream_logits, emotion_logits, domain_logits = model(batch)

        scream_loss = torch.nn.functional.nll_loss(
            scream_logits, batch.y_scream.squeeze()
        )
        emotion_loss = torch.nn.functional.nll_loss(
            emotion_logits, batch.y_emotion.squeeze()
        )
        domain_labels = 1 - batch.domain.squeeze()
        domain_loss = torch.nn.functional.nll_loss(domain_logits, domain_labels)

        total_loss = 0.6 * scream_loss + 0.4 * emotion_loss + 0.3 * domain_loss
        total_loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            scream_logits, _, _ = model(batch)
            pred = scream_logits.argmax(dim=1)
            correct += (pred == batch.y_scream.squeeze()).sum().item()

    acc = correct / len(val_loader.dataset)
    if acc > best_acc:
        best_acc = acc

    print(f"  Epoch {epoch + 1}/20 | Acc: {acc:.2%} | Best: {best_acc:.2%}")

# Save
print("\nSaving...")
os.makedirs("scream_models", exist_ok=True)
torch.save(model.state_dict(), "scream_models/multitask_ggnn.pt")

import json

config = {
    "num_node_features": num_features,
    "hidden_channels": 64,
    "num_layers": 4,
    "scream_accuracy": float(best_acc),
}
with open("scream_models/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n=== DONE === Best: {best_acc:.2%}")
