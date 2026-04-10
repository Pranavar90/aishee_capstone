import os
import sys
import json
import random
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import kagglehub

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Configuration
CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "hidden_channels": 64,
    "num_layers": 4,
    "val_split": 0.15,
}


class ScreamDataset(Dataset):
    """Custom Dataset for Scream Detection."""

    def __init__(self, file_paths, labels, sr=22050, duration=2.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            y, _ = librosa.load(
                self.file_paths[idx], sr=self.sr, duration=self.duration
            )

            if len(y) < self.sr * 0.5:
                return self._get_empty_sample()

            # Extract features (include stress features for better accuracy)
            from src.features import (
                extract_features_from_array,
                build_graph_from_features,
            )

            feats = extract_features_from_array(y, sr=self.sr, include_stress=True)

            if feats is None:
                return self._get_empty_sample()

            g_data = build_graph_from_features(feats)

            if g_data is None:
                return self._get_empty_sample()

            g_data.y_scream = torch.tensor([self.labels[idx]], dtype=torch.long)
            g_data.y_emotion = torch.tensor([0], dtype=torch.long)
            g_data.domain = torch.tensor([0], dtype=torch.long)

            return g_data

        except Exception as e:
            return self._get_empty_sample()

    def _get_empty_sample(self):
        """Return an empty graph sample."""
        x = torch.zeros(10, 29, dtype=torch.float)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        data.y_scream = torch.tensor([0], dtype=torch.long)
        data.y_emotion = torch.tensor([0], dtype=torch.long)
        data.domain = torch.tensor([0], dtype=torch.long)
        return data


def download_dataset_once():
    """
    Downloads the dataset once and saves to local folder.
    Uses kagglehub cache, downloads only if not already cached.
    """
    dataset_name = "whats2000/human-screaming-detection-dataset"
    cache_dir = r"C:\Users\ASUS\.cache\kagglehub\datasets\whats2000\human-screaming-detection-dataset\versions\2"

    # Check if already downloaded
    if os.path.exists(cache_dir):
        print(f"Dataset already cached at: {cache_dir}")
        return cache_dir

    # Download (only happens once)
    print("Downloading dataset (this only happens once)...")
    try:
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def get_local_file_paths(dataset_path):
    """Get file paths from local dataset."""
    scream_files = []
    non_scream_files = []

    for root, dirs, files in os.walk(dataset_path):
        folder_name = os.path.basename(root).lower()

        for file in files:
            if file.lower().endswith(".wav"):
                full_path = os.path.join(root, file)

                # Check folder for labeling
                if "notscreaming" in folder_name or "noise" in folder_name:
                    non_scream_files.append(full_path)
                elif "screaming" in folder_name:
                    scream_files.append(full_path)
                # Fallback: check full path
                elif "notscream" in root.lower():
                    non_scream_files.append(full_path)
                elif "scream" in root.lower():
                    scream_files.append(full_path)

    return scream_files, non_scream_files


def train_cpu():
    """Training function that runs on CPU."""
    print("=" * 60)
    print("SCREAM DETECTION MODEL TRAINING")
    print("=" * 60)

    # Step 1: Download dataset once
    print("\n[STEP 1] Downloading/Loading Dataset...")
    dataset_path = download_dataset_once()

    if not dataset_path:
        print("Failed to load dataset!")
        return

    # Step 2: Get file paths
    print("\n[STEP 2] Scanning Files...")
    scream_files, non_scream_files = get_local_file_paths(dataset_path)
    print(f"Found: {len(scream_files)} screams, {len(non_scream_files)} ambient")

    # Step 3: Balance dataset
    print("\n[STEP 3] Preparing Dataset...")
    max_per_class = min(len(scream_files), len(non_scream_files), 1000)
    scream_files = scream_files[:max_per_class]
    non_scream_files = non_scream_files[:max_per_class]

    all_files = scream_files + non_scream_files
    all_labels = [1] * len(scream_files) + [0] * len(non_scream_files)

    # Shuffle
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)
    all_files = list(all_files)
    all_labels = list(all_labels)

    print(f"Total samples: {len(all_files)}")

    # Step 4: Split dataset
    split_idx = int(len(all_files) * (1 - CONFIG["val_split"]))
    train_files = all_files[:split_idx]
    train_labels = all_labels[:split_idx]
    val_files = all_files[split_idx:]
    val_labels = all_labels[split_idx:]

    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")

    # Step 5: Create datasets and dataloaders
    print("\n[STEP 4] Creating DataLoaders...")
    train_dataset = ScreamDataset(train_files, train_labels)
    val_dataset = ScreamDataset(val_files, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    # Step 6: Build model
    print("\n[STEP 5] Building Model...")
    from src.models import MultiHeadGGNN

    # Get feature dimension from first sample
    sample = train_dataset[0]
    num_features = sample.num_node_features
    print(f"Feature dimension: {num_features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")

    model = MultiHeadGGNN(
        num_node_features=num_features,
        hidden_channels=CONFIG["hidden_channels"],
        num_layers=CONFIG["num_layers"],
        num_emotion_classes=5,
        lstm_hidden=CONFIG["hidden_channels"],
        use_dann=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Step 7: Training
    print("\n[STEP 6] Training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    best_acc = 0
    best_state = None
    patience = 0
    max_patience = 10

    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        train_loss = 0

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
            domain_loss = torch.nn.functional.nll_loss(
                domain_logits, 1 - batch.domain.squeeze()
            )

            total_loss = 0.6 * scream_loss + 0.3 * emotion_loss + 0.1 * domain_loss
            total_loss.backward()
            optimizer.step()

            train_loss += scream_loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                scream_logits, _, _ = model(batch)
                pred = scream_logits.argmax(dim=1)
                correct += (pred == batch.y_scream.squeeze()).sum().item()
                total += batch.y_scream.size(0)

        acc = correct / total if total > 0 else 0

        print(
            f"  Epoch {epoch + 1}/{CONFIG['epochs']} | Loss: {train_loss:.4f} | Acc: {acc:.2%}"
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Step 8: Save model
    print("\n[STEP 7] Saving Model...")
    if best_state:
        model.load_state_dict(best_state)

    os.makedirs("scream_models", exist_ok=True)
    torch.save(model.state_dict(), "scream_models/multitask_ggnn.pt")

    # Save config
    config_data = {
        "num_node_features": num_features,
        "hidden_channels": CONFIG["hidden_channels"],
        "num_layers": CONFIG["num_layers"],
        "scream_accuracy": float(best_acc),
        "training_config": CONFIG,
        "model_type": "MultiHeadGGNN_DANN",
    }

    with open("scream_models/config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE!")
    print(f"Best Accuracy: {best_acc:.2%}")
    print(f"Model saved to: scream_models/multitask_ggnn.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_cpu()
