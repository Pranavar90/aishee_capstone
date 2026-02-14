import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
import torch
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from src import data_loader, features, models, utils

def train_and_optimize():
    print("=== Safety Intelligence Training Pipeline ===")
    
    # 1. Data
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

    # 2. Features
    print("\n[Step 2] Processing Features & Graphs...")
    labels = []
    
    # We will store paths and process them. To avoid OOM or huge wait, 
    # let's limit for PoC if dataset is massive (thousands is okay).
    # The dataset has ~3k files.
    
    # Container for paired data: (graph, svm_vec, label)
    processed_data = []

    def process_batch(file_list, label, max_files=None):
        count = 0
        for fpath in file_list:
            if max_files and count >= max_files:
                break
            try:
                # Extract
                feats = features.extract_features(fpath, sr=22050)
                if feats is not None:
                    # Graph
                    g_data = features.build_graph_from_features(feats)
                    
                    # SVM Vector (Global Mean of MFCCs e.g. first 20 cols)
                    mfccs = feats[:, :20]
                    svm_vec = np.mean(mfccs, axis=0)
                    
                    if g_data is not None:
                         # Assign label to graph for PyG
                        g_data.y = torch.tensor([label], dtype=torch.long)
                        processed_data.append((g_data, svm_vec, label))
                        count += 1
            except Exception as e:
                pass
                # print(f"Error processing {fpath}: {e}")
        return count

    n_scream = process_batch(scream_files, 1)
    n_noise = process_batch(non_scream_files, 0)
    print(f"  - Successfully processed: {len(processed_data)} items")
    
    if len(processed_data) == 0:
        print("No valid data processed.")
        return

    # Unzip
    data_graphs = [d[0] for d in processed_data]
    data_svm = np.array([d[1] for d in processed_data])
    y_labels = np.array([d[2] for d in processed_data])
    
    # Metrics for Class Balance
    print(f"  - Balance: {sum(y_labels)} Positive / {len(y_labels) - sum(y_labels)} Negative")

    # Split
    # Indices
    indices = np.arange(len(y_labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_labels)
    
    X_train_svm, X_test_svm = data_svm[train_idx], data_svm[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]
    
    train_graphs = [data_graphs[i] for i in train_idx]
    test_graphs = [data_graphs[i] for i in test_idx]

    # 3. SVM Optimization
    print("\n[Step 3] Optimizing SVM (Optuna)...")
    
    def objective_svm(trial):
        c = trial.suggest_float("C", 1e-3, 1e2, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        clf = models.ScreamSVM(C=c, gamma=gamma)
        clf.fit(X_train_svm, y_train)
        preds = clf.predict(X_test_svm)
        return accuracy_score(y_test, preds)

    # Suppress optuna logging for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(objective_svm, n_trials=15)
    
    print(f"  - Best SVM Accuracy: {study_svm.best_value:.4f}")
    print(f"  - Best SVM Params: {study_svm.best_params}")
    
    # Train Final SVM
    best_svm = models.ScreamSVM(**study_svm.best_params)
    best_svm.fit(data_svm, y_labels) # Fit on all data

    # 4. GGNN Optimization
    print("\n[Step 4] Optimizing GGNN (Optuna)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        hidden = trial.suggest_int("hidden_channels", 16, 64)
        layers = trial.suggest_int("num_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        
        # DataLoaders
        # Use smaller batch for speed
        loader_tr = DataLoader(train_graphs, batch_size=32, shuffle=True)
        loader_te = DataLoader(test_graphs, batch_size=32)
        
        num_feats = train_graphs[0].num_node_features
        model = models.ScreamGGNN(num_node_features=num_feats, hidden_channels=hidden, num_layers=layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Fast epochs for tuning
        for _ in range(5):
            train_ggnn_epoch(model, loader_tr, optimizer)
            
        return eval_ggnn(model, loader_te)

    study_ggnn = optuna.create_study(direction="maximize")
    study_ggnn.optimize(objective_ggnn, n_trials=10)
    
    print(f"  - Best GGNN Accuracy: {study_ggnn.best_value:.4f}")
    print(f"  - Best GGNN Params: {study_ggnn.best_params}")
    
    # Train Final GGNN
    print("  - Training final GGNN model...")
    full_loader = DataLoader(data_graphs, batch_size=32, shuffle=True)
    best_p = study_ggnn.best_params
    
    final_ggnn = models.ScreamGGNN(
        num_node_features=data_graphs[0].num_node_features,
        hidden_channels=best_p['hidden_channels'],
        num_layers=best_p['num_layers']
    ).to(device)
    
    opt_final = torch.optim.Adam(final_ggnn.parameters(), lr=best_p['lr'])
    
    # Train for more epochs
    for ep in range(15):
        loss = train_ggnn_epoch(final_ggnn, full_loader, opt_final)
        # print(f"    Epoch {ep+1}: Loss {loss:.4f}")
        
    # 5. Serialization
    print("\n[Step 5] Saving Models...")
    utils.save_models(best_svm, final_ggnn, output_dir='scream_models')
    
    # Save config for the app
    import json
    config = {
        "num_node_features": data_graphs[0].num_node_features,
        "hidden_channels": best_p['hidden_channels'],
        "num_layers": best_p['num_layers'],
        "svm_accuracy": float(study_svm.best_value),
        "ggnn_accuracy": float(study_ggnn.best_value),
        "total_samples": len(processed_data),
        "scream_samples": n_scream,
        "non_scream_samples": n_noise
    }
    with open(os.path.join('scream_models', 'config.json'), 'w') as f:
        json.dump(config, f)
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_and_optimize()
