import joblib
import torch
import os

def save_models(svm_model, ggnn_model, output_dir='.'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    svm_path = os.path.join(output_dir, 'scream_svm.pkl')
    ggnn_path = os.path.join(output_dir, 'scream_ggnn.pt')
    
    # Save SVM
    joblib.dump(svm_model, svm_path)
    print(f"SVM model saved to {svm_path}")
    
    # Save GGNN
    torch.save(ggnn_model.state_dict(), ggnn_path)
    print(f"GGNN model saved to {ggnn_path}")

def load_models(model_dir='.'):
    svm_path = os.path.join(model_dir, 'scream_svm.pkl')
    ggnn_path = os.path.join(model_dir, 'scream_ggnn.pt')
    
    svm_model = None
    if os.path.exists(svm_path):
        svm_model = joblib.load(svm_path)
    
    # GGNN loading requires initializing the class structure first with correct args.
    # We will assume default or saved args. For simplicity, we return the path 
    # and let the caller load state_dict into their model instance,
    # or we construct it if we know the hyperparameters.
    # For now, we'll return the path for GGNN or a loaded dict.
    
    return svm_model, ggnn_path
