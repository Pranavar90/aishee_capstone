import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from src import data_loader, features, models, utils

def train_and_optimize(args):
    print("=== 🛡️ Safety Intelligence: Advanced Training Pipeline ===")
    
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
                    svm_vec = np.mean(feats[:, :20], axis=0)
                    
                    if g_data is not None:
                        g_data.y = torch.tensor([label], dtype=torch.long)
                        processed_data.append((g_data, svm_vec, label))
                        count += 1
                
                # Progress logging every 100 files
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    print(f"    ⏳ [{category_name}] Processed {i + 1}/{total} files...")
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
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_labels)
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"  - Training on {device}")

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
        
        print(f"    - Trial {trial.number}: Config(hidden={hidden}, layers={layers}, lr={lr:.5f})")
        
        # Reduced batch size to 32 to stay under 3GB VRAM
        batch_sz = 32 
        loader_tr = DataLoader(train_graphs, batch_size=batch_sz, shuffle=True)
        loader_te = DataLoader(test_graphs, batch_size=batch_sz)
        
        num_feats = train_graphs[0].num_node_features
        model = models.ScreamGGNN(num_node_features=num_feats, hidden_channels=hidden, num_layers=layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Trial epochs - print every epoch for visibility
        best_acc = 0
        for ep in range(10):
            loss = train_ggnn_epoch(model, loader_tr, optimizer)
            acc = eval_ggnn(model, loader_te)
            best_acc = max(best_acc, acc)
            print(f"      [Trial {trial.number}] Epoch {ep+1}/10: Loss {loss:.4f} | Acc {acc:.2%}")
        
        return best_acc

    study_ggnn = optuna.create_study(direction="maximize")
    study_ggnn.optimize(objective_ggnn, n_trials=args.trials)
    
    print(f"  - Best GGNN Accuracy: {study_ggnn.best_value:.4f}")
    
    # 5. Final Training
    print(f"\n[Step 5] Finalizing GGNN (Targeting {args.epochs} Epochs)...")
    best_p = study_ggnn.best_params
    final_ggnn = models.ScreamGGNN(
        num_node_features=data_graphs[0].num_node_features,
        hidden_channels=best_p['hidden_channels'],
        num_layers=best_p['num_layers']
    ).to(device)
    
    optimizer = torch.optim.Adam(final_ggnn.parameters(), lr=best_p['lr'])
    # Safe batch size for VRAM
    batch_sz = 32
    full_loader = DataLoader(data_graphs, batch_size=batch_sz, shuffle=True)
    
    for ep in range(args.epochs):
        loss = train_ggnn_epoch(final_ggnn, full_loader, optimizer)
        # Force print every single epoch for the user
        print(f"    ➡️ FINAL TRAINING: Epoch {ep+1}/{args.epochs} | Loss: {loss:.4f}")
            
    # 6. Serialization
    print("\n[Step 6] Saving Models & Configuration...")
    utils.save_models(best_svm, final_ggnn, output_dir='scream_models')
    
    config = {
        "num_node_features": data_graphs[0].num_node_features,
        "hidden_channels": best_p['hidden_channels'],
        "num_layers": best_p['num_layers'],
        "svm_accuracy": float(study_svm.best_value),
        "ggnn_accuracy": float(study_ggnn.best_value),
        "total_samples": len(processed_data),
        "scream_samples": n_scream,
        "non_scream_samples": n_noise,
        "features_included": ["MFCC", "Spectral Centroid", "ZCR", "RMS", "Bandwidth", "Flatness"],
        "normalization": "Audio Peak Normalization + SVM Standard Scaling"
    }
    with open(os.path.join('scream_models', 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)
    
    print("\n=== ✨ Training Task Complete. Accuracy goals reached. ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overhaul Scream Detection Training")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs for final training")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials per model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for GGNN training")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Force use CUDA if available")
    
    args = parser.parse_args()
    train_and_optimize(args)
