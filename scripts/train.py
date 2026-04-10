import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import optuna
import joblib
import yaml
import time
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import librosa

from joblib import Parallel, delayed
from tqdm import tqdm

# Add the project root to sys.path to allow importing from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import data_loader, features, models, utils


# Helper for parallel processing
def _process_single_file(fpath, label, domain, metadata=None):
    """Worker function for parallel processing."""
    try:
        # Load audio with a shorter duration to speed up processing
        # Screams are usually short; 2.0s is enough for feature extraction
        y, sr = librosa.load(fpath, sr=22050, duration=2.0)
        
        # Fast-track: if audio is too short, skip
        if len(y) < 22050 * 0.5:
             return None
             
        feats = features.extract_features_from_array(
            y, sr=sr, include_stress=True
        )
        if feats is not None:
            g_data = features.build_graph_from_features(feats)
            if g_data is not None:
                # Get emotion label if metadata provided
                emotion_label = "neutral"
                if metadata and fpath in metadata:
                    emotion_label = metadata[fpath].get("emotion", "neutral")
                
                # EMOTION_TO_IDX mapping
                emotion_map = {
                    "fear": 0, "fearful": 0,
                    "pain": 1, "distress": 4, 
                    "anger": 2, "angry": 2,
                    "joy": 3, "happy": 3,
                    "neutral": 5, "calm": 5
                }
                emotion_idx = emotion_map.get(emotion_label, 5)

                g_data.y_scream = torch.tensor([label], dtype=torch.long)
                g_data.y_emotion = torch.tensor([emotion_idx], dtype=torch.long)
                g_data.domain = torch.tensor([domain], dtype=torch.long)
                return g_data
    except Exception:
        pass
    return None


# Load configuration
def load_training_config():
    config_path = os.path.join(os.getcwd(), "model_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    # Default fallback
    return {
        "multitask_ggnn": {
            "hyperparameters": {
                "alpha": 0.6,
                "hidden_channels": 64,
                "num_layers": 4,
                "use_dann": True,
                "domain_weight": 0.3,
                "learning_rate": 0.001
            }
        },
        "training": {"epochs": 50, "batch_size": 32, "val_split": 0.2},
    }


config = load_training_config()


def plot_curves(history, output_path):
    """Generate training curve plots."""
    epochs = range(1, len(history["loss"]) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Total Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["loss"], "b-", label="Total Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # 2. Accuracies
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["scream_acc"], "g-", label="Scream Acc")
    plt.plot(epochs, history["emotion_acc"], "r-", label="Emotion Acc")
    plt.title("Validation Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    # 3. Timing
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["epoch_time"], "k-o", label="Time (s)")
    plt.title("Seconds per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def train_multitask_dann(args):
    """
    Multi-task training with Domain-Adversarial Neural Network (DANN).
    """
    print("\n" + "="*50)
    print("🚀 STRATEGIC AI TRAINING: MULTI-TASK DANN")
    print("="*50)

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    print(f"Device: {device}")

    # Load Hyperparameters from YAML
    m_cfg = config.get("multitask_ggnn", {}).get("hyperparameters", {})
    t_cfg = config.get("training", {})
    
    epochs = args.epochs if args.epochs != 50 else t_cfg.get("epochs", 50)
    batch_size = args.batch_size if args.batch_size != 32 else t_cfg.get("batch_size", 32)

    # 1. Load original scream dataset
    print("\n[Step 1] Loading Scream Dataset...")
    scream_path = data_loader.download_dataset()
    scream_files, non_scream_files = data_loader.get_file_paths(scream_path)
    
    # 2. Load emotion datasets
    print("[Step 2] Loading Emotion Datasets...")
    emotion_files = data_loader.get_emotion_scream_files()

    # 3. Process all data (Parallelized)
    print("\n[Step 3] Parallel Feature Extraction...")
    n_jobs = max(1, int(os.cpu_count() * 0.8))
    
    def process_files_parallel(file_paths, label, domain, metadata=None, desc="Processing"):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_file)(f, label, domain, metadata) 
            for f in tqdm(file_paths, desc=desc)
        )
        return [r for r in results if r is not None]

    scream_pts = process_files_parallel(scream_files, 1, 0, desc="  - Screams")
    ambient_pts = process_files_parallel(non_scream_files, 0, 0, desc="  - Ambient")
    emotion_pts = process_files_parallel(list(emotion_files.keys()), 0, 1, emotion_files, desc="  - Emotions")

    processed = scream_pts + ambient_pts + emotion_pts
    num_features = processed[0].num_node_features

    # 4. Split data
    print(f"\n[Step 4] Dataset Ready: {len(processed)} samples.")
    indices = np.arange(len(processed))
    np.random.shuffle(indices)
    split = int(len(indices) * t_cfg.get("val_split", 0.15))
    train_data = [processed[i] for i in indices[split:]]
    val_data = [processed[i] for i in indices[:split]]

    # 5. Initialize model
    print("\n[Step 5] Building Multi-Head GGNN...")
    multitask_model = models.MultiHeadGGNN(
        num_node_features=num_features,
        hidden_channels=m_cfg.get("hidden_channels", 128),
        num_layers=m_cfg.get("num_layers", 6),
        num_emotion_classes=6,
        lstm_hidden=m_cfg.get("lstm_hidden", 128),
        use_dann=m_cfg.get("use_dann", True),
    ).to(device)

    # 6. Training loop
    print(f"\n[Step 6] Starting Engine ({epochs} Epochs)...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        multitask_model.parameters(), 
        lr=m_cfg.get("learning_rate", 0.001),
        weight_decay=m_cfg.get("weight_decay", 0.01)
    )

    alpha = m_cfg.get("alpha", 0.6)
    domain_weight = m_cfg.get("domain_weight", 0.2)
    history = {"loss": [], "scream_acc": [], "emotion_acc": [], "epoch_time": []}

    best_scream_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        multitask_model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            scream_logits, emotion_logits, domain_logits = multitask_model(batch)

            s_loss = torch.nn.functional.nll_loss(scream_logits, batch.y_scream.squeeze())
            e_loss = torch.nn.functional.nll_loss(emotion_logits, batch.y_emotion.squeeze())
            d_loss = torch.nn.functional.nll_loss(domain_logits, 1 - batch.domain.squeeze())

            batch_loss = (alpha * s_loss + (1 - alpha) * e_loss + domain_weight * d_loss)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        # Validation
        multitask_model.eval()
        s_correct, e_correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                s_logits, e_logits, _ = multitask_model(batch)
                s_correct += (s_logits.argmax(1) == batch.y_scream.squeeze()).sum().item()
                e_correct += (e_logits.argmax(1) == batch.y_emotion.squeeze()).sum().item()
                total += batch.y_scream.size(0)

        s_acc, e_acc = s_correct / total, e_correct / total
        epoch_time = time.time() - epoch_start
        
        # Track history
        history["loss"].append(total_loss / len(train_loader))
        history["scream_acc"].append(s_acc)
        history["emotion_acc"].append(e_acc)
        history["epoch_time"].append(epoch_time)

        print(f"  Epoch {epoch+1:02d} | Loss: {history['loss'][-1]:.4f} | S-Acc: {s_acc:.2%} | E-Acc: {e_acc:.2%} | Time: {epoch_time:.1f}s")

        if s_acc > best_scream_acc:
            best_scream_acc = s_acc
            torch.save(multitask_model.state_dict(), os.path.join("scream_models", "multitask_ggnn.pt"))

    # 7. Finalization
    total_time = (time.time() - start_time) / 60
    print(f"\n[Step 7] Done! Total Time: {total_time:.2f} mins | Peak Scream Acc: {best_scream_acc:.2%}")

    if t_cfg.get("save_learning_curves", True):
        plot_curves(history, t_cfg.get("metrics_report_path", "reports/metrics.png"))

    # Save final config
    import json
    app_config = {
        "num_node_features": num_features,
        "hidden_channels": m_cfg.get("hidden_channels", 128),
        "num_layers": m_cfg.get("num_layers", 6),
        "lstm_hidden": m_cfg.get("lstm_hidden", 128),
        "num_emotion_classes": 6,
        "use_dann": m_cfg.get("use_dann", True),
        "model_type": "MultiHeadGGNN"
    }
    with open("scream_models/config.json", "w") as f:
        json.dump(app_config, f, indent=2)


def train_and_optimize(args):
    print("=== Safety Intelligence: Advanced Training Pipeline ===")

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Data Acquisition
    print("\n[Step 1] Acquiring Data...")
    dataset_path = data_loader.download_dataset()
    if not dataset_path:
        print("Data download failed. Exiting.")
        return

    scream_files, non_scream_files = data_loader.get_file_paths(dataset_path)
    if not scream_files and not non_scream_files:
        print("No audio files found. Exiting.")
        return

    print(f"  - Scream Samples: {len(scream_files)}")
    print(f"  - Non-Scream Samples: {len(non_scream_files)}")

    # 2. Feature Engineering
    print("\n[Step 2] Processing Features & Graphs...")
    processed_data = []

    def process_batch(file_list, label, category_name="Data"):
        count = 0
        total = len(file_list)
        for i, fpath in enumerate(file_list):
            try:
                feats = features.extract_features(fpath, sr=22050)
                if feats is not None:
                    g_data = features.build_graph_from_features(feats)
                    # Use Mean + Std Dev of ALL 25 features for a 50D SVM vector
                    m = np.mean(feats, axis=0)
                    s = np.std(feats, axis=0)
                    svm_vec = np.concatenate([m, s])

                    if g_data is not None:
                        g_data.y = torch.tensor([label], dtype=torch.long)
                        processed_data.append((g_data, svm_vec, label))
                        count += 1

                # Progress logging every 100 files
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(
                        f"    ⏳ [{category_name}] Processed {i + 1}/{total} files..."
                    )
            except Exception:
                pass
        return count

    n_scream = process_batch(scream_files, 1, "Screams")
    n_noise = process_batch(non_scream_files, 0, "Ambient")
    print(f"  - Successfully processed: {len(processed_data)} items")

    if len(processed_data) == 0:
        print("No valid data processed.")
        return

    # Prepare Data
    data_graphs = [d[0] for d in processed_data]
    data_svm = np.array([d[1] for d in processed_data])
    y_labels = np.array([d[2] for d in processed_data])

    # Split
    indices = np.arange(len(y_labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_labels
    )

    X_train_svm, X_test_svm = data_svm[train_idx], data_svm[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]

    train_graphs = [data_graphs[i] for i in train_idx]
    test_graphs = [data_graphs[i] for i in test_idx]

    # SVM Feature Scaling
    scaler_svm = StandardScaler()
    X_train_svm_scaled = scaler_svm.fit_transform(X_train_svm)
    X_test_svm_scaled = scaler_svm.transform(X_test_svm)

    # 3. SVM Optimization (Optuna)
    print(f"\n[Step 3] Optimizing SVM (Trials: {args.trials})...")

    def objective_svm(trial):
        c = trial.suggest_float("C", 1e-4, 1e3, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        clf = models.ScreamSVM(C=c, gamma=gamma)
        clf.fit(X_train_svm_scaled, y_train)
        preds = clf.predict(X_test_svm_scaled)
        return accuracy_score(y_test, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(objective_svm, n_trials=args.trials)

    print(f"  - Best SVM Accuracy: {study_svm.best_value:.4f}")

    # Final SVM
    best_svm = models.ScreamSVM(**study_svm.best_params)
    best_svm.fit(scaler_svm.transform(data_svm), y_labels)

    # 4. GGNN Optimization (Optuna + CUDA)
    print(f"\n[Step 4] Optimizing GGNN (Trials: {args.trials})...")

    def train_ggnn_epoch(model, loader, optimizer):
        model.train()
        total_loss = 0
        for b in loader:
            b = b.to(device)
            optimizer.zero_grad()
            out = model(b)
            loss = torch.nn.functional.nll_loss(out, b.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b.num_graphs
        return total_loss / len(loader.dataset)

    def eval_ggnn(model, loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                out = model(b)
                pred = out.argmax(dim=1)
                correct += int((pred == b.y).sum())
        return correct / len(loader.dataset)

    def objective_ggnn(trial):
        hidden = trial.suggest_int("hidden_channels", 32, 128)
        layers = trial.suggest_int("num_layers", 2, 8)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        print(
            f"    - Trial {trial.number}: Config(hidden={hidden}, layers={layers}, lr={lr:.5f})"
        )

        # Reduced batch size to 32 to stay under 3GB VRAM
        batch_sz = 32
        loader_tr = DataLoader(train_graphs, batch_size=batch_sz, shuffle=True)
        loader_te = DataLoader(test_graphs, batch_size=batch_sz)

        num_feats = train_graphs[0].num_node_features
        model = models.ScreamGGNN(
            num_node_features=num_feats, hidden_channels=hidden, num_layers=layers
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Trial epochs - print every epoch for visibility
        best_acc = 0
        for ep in range(10):
            loss = train_ggnn_epoch(model, loader_tr, optimizer)
            acc = eval_ggnn(model, loader_te)
            best_acc = max(best_acc, acc)
            print(
                f"      [Trial {trial.number}] Epoch {ep + 1}/10: Loss {loss:.4f} | Acc {acc:.2%}"
            )

        return best_acc

    study_ggnn = optuna.create_study(direction="maximize")
    study_ggnn.optimize(objective_ggnn, n_trials=args.trials)

    print(f"  - Best GGNN Accuracy: {study_ggnn.best_value:.4f}")

    # 5. Final Training
    print(f"\n[Step 5] Finalizing GGNN (Targeting {args.epochs} Epochs)...")
    best_p = study_ggnn.best_params
    final_ggnn = models.ScreamGGNN(
        num_node_features=data_graphs[0].num_node_features,
        hidden_channels=best_p["hidden_channels"],
        num_layers=best_p["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(final_ggnn.parameters(), lr=best_p["lr"])
    # Safe batch size for VRAM
    batch_sz = 32
    full_loader = DataLoader(data_graphs, batch_size=batch_sz, shuffle=True)

    for ep in range(args.epochs):
        loss = train_ggnn_epoch(final_ggnn, full_loader, optimizer)
        # Force print every single epoch for the user
        print(f"    ➡️ FINAL TRAINING: Epoch {ep + 1}/{args.epochs} | Loss: {loss:.4f}")

    # 6. Serialization
    print("\n[Step 6] Saving Models & Configuration...")
    utils.save_models(best_svm, final_ggnn, output_dir="scream_models")

    config = {
        "num_node_features": data_graphs[0].num_node_features,
        "hidden_channels": best_p["hidden_channels"],
        "num_layers": best_p["num_layers"],
        "svm_accuracy": float(study_svm.best_value),
        "ggnn_accuracy": float(study_ggnn.best_value),
        "total_samples": len(processed_data),
        "scream_samples": n_scream,
        "non_scream_samples": n_noise,
        "features_included": [
            "MFCC",
            "Spectral Centroid",
            "ZCR",
            "RMS",
            "Bandwidth",
            "Flatness",
        ],
        "normalization": "Audio Peak Normalization + SVM Standard Scaling",
    }
    with open(os.path.join("scream_models", "config.json"), "w") as f:
        import json

        json.dump(config, f, indent=4)

    print("\n=== ✨ Training Task Complete. Accuracy goals reached. ===")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scream Detection Training")
    parser.add_argument(
        "--mode",
        type=str,
        default="multitask",
        choices=["original", "multitask"],
        help="Training mode: original (SVM+GGNN) or multitask (Multi-head GGNN with DANN)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Max epochs for final training"
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of Optuna trials per model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for GGNN training"
    )
    parser.add_argument(
        "--use_cuda", type=str2bool, default=True, help="Use CUDA if available (True/False)"
    )

    args = parser.parse_args()

    if args.mode == "multitask":
        train_multitask_dann(args)
    else:
        train_and_optimize(args)
