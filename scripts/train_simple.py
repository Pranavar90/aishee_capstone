import os
import numpy as np
import torch
import yaml
import librosa
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
# Add project root to sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import features, models


def load_config():
    config_path = os.path.join(os.getcwd(), "model_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {"model": {"alpha": 0.6, "hidden_channels": 64, "num_layers": 4}}


config = load_config()


def process_audio_file(fpath, label, sr=22050):
    """Process a single audio file into graph data."""
    try:
        y, sr = librosa.load(fpath, sr=sr, duration=3.0)
        if len(y) < sr * 0.5:
            return None

        feats = features.extract_features_from_array(y, sr=sr, include_stress=True)
        if feats is None:
            return None

        g_data = features.build_graph_from_features(feats)
        if g_data is None:
            return None

        g_data.y_scream = torch.tensor([label], dtype=torch.long)
        g_data.y_emotion = torch.tensor([0], dtype=torch.long)  # neutral
        g_data.domain = torch.tensor([0], dtype=torch.long)

        return g_data
    except Exception as e:
        return None


def train_with_scream_data():
    print("=== Multi-Task Training with Scream Dataset ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from src import data_loader

    print("\n[Step 1] Loading dataset paths...")
    dataset_path = data_loader.download_dataset()
    if not dataset_path:
        print("Failed to load dataset")
        return

    scream_files, non_scream_files = data_loader.get_file_paths(dataset_path)
    print(f"Found: Scream={len(scream_files)}, Ambient={len(non_scream_files)}")

    print("\n[Step 2] Processing audio files...")
    data_list = []

    all_files = scream_files + non_scream_files[: len(scream_files)]  # Balance classes
    labels = [1] * len(scream_files) + [0] * len(scream_files)

    for i, (fpath, label) in enumerate(zip(all_files, labels)):
        g_data = process_audio_file(fpath, label)
        if g_data:
            data_list.append(g_data)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_files)}...")

    print(f"  Total processed: {len(data_list)}")

    if len(data_list) == 0:
        print("No data processed!")
        return

    num_features = data_list[0].num_node_features
    print(f"  Feature dimension: {num_features}")

    print("\n[Step 3] Splitting data...")
    np.random.shuffle(data_list)
    split = int(len(data_list) * 0.8)
    train_data = data_list[:split]
    val_data = data_list[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    print("\n[Step 4] Initializing model...")
    model_cfg = config.get("model", {})
    model = models.MultiHeadGGNN(
        num_node_features=num_features,
        hidden_channels=model_cfg.get("hidden_channels", 64),
        num_layers=model_cfg.get("num_layers", 4),
        num_emotion_classes=5,
        lstm_hidden=64,
        use_dann=True,
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n[Step 5] Training...")
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    alpha = model_cfg.get("alpha", 0.6)
    domain_weight = model_cfg.get("domain_weight", 0.3)

    best_acc = 0

    for epoch in range(30):
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

            total_loss = (
                alpha * scream_loss
                + (1 - alpha) * emotion_loss
                + domain_weight * domain_loss
            )
            total_loss.backward()
            optimizer.step()

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

        print(f"  Epoch {epoch + 1}/30 | Acc: {acc:.2%} | Best: {best_acc:.2%}")

    print("\n[Step 6] Saving model...")
    os.makedirs("scream_models", exist_ok=True)
    torch.save(model.state_dict(), "scream_models/multitask_ggnn.pt")

    # Save config
    import json

    save_config = {
        "num_node_features": num_features,
        "hidden_channels": model_cfg.get("hidden_channels", 64),
        "num_layers": model_cfg.get("num_layers", 4),
        "scream_accuracy": float(best_acc),
    }
    with open("scream_models/config.json", "w") as f:
        json.dump(save_config, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Best Scream Accuracy: {best_acc:.2%}")

    return model


if __name__ == "__main__":
    train_with_scream_data()
