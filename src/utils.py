"""
src/utils.py — Model serialisation helpers.
"""

import os
import joblib
import torch


def save_models(svm_model, ggnn_model, output_dir="scream_models"):
    """Save SVM (joblib) and GGNN state-dict to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    svm_path  = os.path.join(output_dir, "scream_svm.pkl")
    ggnn_path = os.path.join(output_dir, "multitask_ggnn.pt")
    joblib.dump(svm_model, svm_path)
    torch.save(ggnn_model.state_dict(), ggnn_path)
    print(f"Saved: {svm_path}, {ggnn_path}")


def load_models(model_dir="scream_models"):
    """Return dict of paths to saved artefacts."""
    return {
        "svm":    os.path.join(model_dir, "scream_svm.pkl"),
        "ggnn":   os.path.join(model_dir, "multitask_ggnn.pt"),
        "config": os.path.join(model_dir, "config.json"),
    }
