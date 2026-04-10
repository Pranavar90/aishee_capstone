import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import models, features, data_loader

def run_evaluation():
    print("="*60)
    print("🛸 STRATEGIC MODEL EVALUATION PROTOTYPE")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "scream_models"
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "multitask_ggnn.pt")

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}. Please train first.")
        return

    # 1. Load Config and Model
    import json
    with open(config_path, "r") as f:
        cfg = json.load(f)

    print(f"📦 Loading {cfg['model_type']}...")
    model = models.MultiHeadGGNN(
        num_node_features=cfg["num_node_features"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        num_emotion_classes=cfg.get("num_emotion_classes", 6)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Prepare Test Data
    print("\n[Step 1] Preparing hidden test slice...")
    path = data_loader.download_dataset()
    scream_files, ambient_files = data_loader.get_file_paths(path)
    
    # Take a small representative sample for evaluation (200 each)
    test_files = scream_files[-200:] + ambient_files[-200:]
    test_labels = [1]*len(scream_files[-200:]) + [0]*len(ambient_files[-200:])

    processed_test = []
    print("  - Extracting features...")
    for f, l in tqdm(zip(test_files, test_labels), total=len(test_files)):
        try:
            feats = features.extract_features(f, sr=22050)
            if feats is not None:
                g_data = features.build_graph_from_features(feats)
                g_data.y_scream = torch.tensor([l], dtype=torch.long)
                processed_test.append(g_data)
        except:
            continue

    loader = DataLoader(processed_test, batch_size=32)

    # 3. Inference
    print("\n[Step 2] Running Inference...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            logits, _, _ = model(batch)
            probs = torch.exp(logits)[:, 1] # Probability of class 1 (Scream)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y_scream.cpu().numpy())

    # 4. Metrics & Visualization
    print("\n[Step 3] Generating Reports...")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ambient', 'Scream'], yticklabels=['Ambient', 'Scream'])
    plt.title("Scream Detection Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # PR Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='purple', lw=2)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.fill_between(recall, precision, alpha=0.2, color='purple')

    report_path = "reports/evaluation_results.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(report_path)
    print(f"✅ Visual report saved to: {report_path}")

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(all_labels, all_preds, target_names=['Ambient', 'Scream']))
    
    print("\n" + "="*60)
    print("🚀 EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
