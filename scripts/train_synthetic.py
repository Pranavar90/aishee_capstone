import os
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from src import features, models


# Load config
def load_config():
    config_path = os.path.join(os.getcwd(), "model_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {"model": {"alpha": 0.6, "hidden_channels": 64, "num_layers": 4}}


config = load_config()


# Generate synthetic audio features
def generate_synthetic_data(n_samples=500, n_features=29):
    """Generate synthetic graph data for training."""
    data_list = []

    for i in range(n_samples):
        # Random feature matrix (T frames, F features)
        n_frames = np.random.randint(20, 50)
        feat_matrix = np.random.randn(n_frames, n_features).astype(np.float32)

        # Build graph
        x = torch.tensor(feat_matrix, dtype=torch.float)

        # Create edges: t -> t+1
        if n_frames > 1:
            source = torch.arange(0, n_frames - 1, dtype=torch.long)
            target = torch.arange(1, n_frames, dtype=torch.long)
            edge_index = torch.stack([source, target])
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Labels: scream (1) or ambient (0)
        is_scream = i < n_samples // 2
        y_scream = 1 if is_scream else 0

        # Emotion: random 0-4 (fear, pain, anger, joy, distress)
        y_emotion = np.random.randint(0, 5)

        # Domain: 0 = real dataset, 1 = emotion datasets
        domain = 0 if is_scream else (1 if np.random.random() > 0.5 else 0)

        data = Data(x=x, edge_index=edge_index)
        data.y_scream = torch.tensor([y_scream], dtype=torch.long)
        data.y_emotion = torch.tensor([y_emotion], dtype=torch.long)
        data.domain = torch.tensor([domain], dtype=torch.long)

        data_list.append(data)

    return data_list


def train_simple():
    print("=== Synthetic Multi-Task Training ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate data
    print("\n[Step 1] Generating synthetic data...")
    data_list = generate_synthetic_data(500)
    print(f"Generated {len(data_list)} samples")

    # Split
    split = int(len(data_list) * 0.8)
    train_data = data_list[:split]
    val_data = data_list[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Model
    print("\n[Step 2] Initializing model...")
    num_features = train_data[0].num_node_features
    print(f"Feature dimension: {num_features}")

    model_cfg = config.get("model", {})
    model = models.MultiHeadGGNN(
        num_node_features=num_features,
        hidden_channels=model_cfg.get("hidden_channels", 64),
        num_layers=model_cfg.get("num_layers", 4),
        num_emotion_classes=5,
        lstm_hidden=64,
        use_dann=True,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print("\n[Step 3] Training...")
    from torch_geometric.loader import DataLoader

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

        # Validation
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

        print(f"Epoch {epoch + 1}/30 | Scream Acc: {acc:.2%} | Best: {best_acc:.2%}")

    # Save
    print("\n[Step 4] Saving model...")
    os.makedirs("scream_models", exist_ok=True)
    torch.save(model.state_dict(), "scream_models/multitask_ggnn.pt")
    print("Saved to scream_models/multitask_ggnn.pt")

    print(f"\n=== Training Complete ===")
    print(f"Best accuracy: {best_acc:.2%}")

    return model


if __name__ == "__main__":
    train_simple()
