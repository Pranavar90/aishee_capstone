# 🛡️ Safety Intelligence: Human Scream Detection Framework

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Geometric-red" alt="PyTorch Geometric">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Prototype-orange" alt="Status">
</div>

## 📖 Overview

**Safety Intelligence** is a modular framework designed for real-time acoustic event detection, specifically focused on identifying **Human Screams** in diverse environments. The system leverages a hybrid ensemble approach combining **Statistical Machine Learning (SVM)** and **Graph Neural Networks (GGNN)** to distinguish genuine distress signals from background noise and conversational speech with high precision.

This repository contains the complete pipeline:
1.  **Research & Training**: Automated data acquisition, advanced feature extraction, and hyperparameter optimization.
2.  **Inference Engine**: A real-time processing core capable of analyzing audio streams.
3.  **Interactive Dashboard**: A strictly tactical, dark-mode web application for monitoring and alerts.

---

## 🏗️ System Architecture

The core of the framework is an **Ensemble Model** that aggregates predictions from two distinct architectural paradigms to ensure robustness.

```mermaid
graph TD
    A[Audio Input (Mic/File)] --> B[Preprocessing & Feature Extraction]
    B --> C{Feature Split}
    
    subgraph SVM_Pipeline [Statistical Path]
        C -->|Global Avg MFCCs| D[SVM (RBF Kernel)]
        D --> E[Probability Score]
    end
    
    subgraph GGNN_Pipeline [Structural Path]
        C -->|Node Features (T frames)| F[Graph Construction]
        F -->|Temporal Graph| G[Gated Graph Neural Network]
        G --> H[Probability Score]
    end
    
    E --> I[Ensemble Aggregator]
    H --> I
    I --> J[Final Decision (Scream/Non-Scream)]
    J --> K[Interactive Dashboard]
```

### 1. Feature Engineering
We transform raw audio waveforms into rich feature representations:
*   **MFCCs (Mel-Frequency Cepstral Coefficients)**: 20 coefficients capturing the timbral texture of the sound.
*   **Spectral Centroid**: Measures the "brightness" of the sound, useful for high-pitched screams.
*   **Zero Crossing Rate (ZCR)**: Indicates the noisiness of the signal.
*   **RMS Energy**: Represents the loudness/intensity.

**Frame Size**: 20ms windows (Nodes in the graph).

### 2. Graph Construction (Temporal Adjacency)
Unlike traditional CNNs that treat audio as images (spectrograms), we model audio as a **Temporal Graph**:
*   **Nodes**: Individual 20ms audio frames containing the feature vector $X_t$.
*   **Edges**: Directed edges connecting frame $t$ to $t+1$, representing the flow of time.
*   **Goal**: This structure allows the GGNN (Gated Graph Neural Network) to pass messages along the temporal dimension, effectively capturing the *evolution* of a scream (onset -> sustain -> decay).

### 3. Model Architectures
*   **Support Vector Machine (SVM)**:
    *   **Input**: Global average of MFCCs over the entire clip.
    *   **Kernel**: Radial Basis Function (RBF) to capture non-linear decision boundaries.
    *   **Role**: Acts as a robust baseline detector for general timbral characteristics.

*   **Gated Graph Neural Network (GGNN)**:
    *   **Input**: The constructed temporal graph.
    *   **Mechanism**: Uses Gated Recurrent Units (GRUs) within the message-passing framework to update node states.
    *   **Role**: Captures complex temporal dependencies and identifying the specific "shriek" pattern that distinguishes screams from loud static noise.

---

## 📂 Project Structure

```
e:/bro_capstone/
├── .streamlit/             # Streamlit configuration (Dark Mode theme)
│   └── config.toml
├── src/                    # Source code package
│   ├── __init__.py
│   ├── data_loader.py      # Kaggle dataset downloader & crawler
│   ├── features.py         # MFCC/Spectral extraction & Graph building
│   ├── models.py           # PyTorch GGNN & Sklearn SVM definitions
│   └── utils.py            # Model serialization & loading tools
├── app.py                  # Real-time Streamlit Dashboard
├── train.py                # Main Training Script (Data -> Train -> Save)
├── requirements.txt        # Python dependencies
└── README.md               # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.8+**
*   **CUDA Toolkit** (Optional, for GPU acceleration with PyTorch)
*   **Microphone** (For live demo)

### 1. Installation

Clone the repository and install the required packages:

```bash
git clone <repository_url>
cd bro_capstone
pip install -r requirements.txt
```

### 2. Model Training

The system uses a dedicated training script to download data, extract features, and optimize models.

1.  Run the training script:
    ```bash
    python train.py
    ```

    This will:
    *   Download the **Human Screaming Detection Dataset** (via `kagglehub` or CLI fallback).
    *   Process audio files into feature vectors and graphs.
    *   Optimize hyperparameters using **Optuna**.
    *   Save the best models to the `scream_models/` directory.

    *Note: Training may take some time depending on your hardware.*

### 3. Running the Dashboard

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## 🎮 Usage Guide

### Interface Controls
*   **Sensitivity Slider**: Adjusts the decision threshold (0.0 - 1.0). Lower values make the system more sensitive but prone to false positives. Higher values require a clearer "scream" signal.
*   **Live Monitor**: Visualizes the audio feed and connection status.
*   **Spectral Analysis**: Displays a real-time spectrogram of the incoming audio buffer.

### Status Indicators
*   <span style="color:green">**🟢 Monitoring (Silence)**</span>: No significant audio detected.
*   <span style="color:orange">**🗣️ Talking / Ambient**</span>: Audio detected but classified as non-scream (e.g., speech, background noise).
*   <span style="color:red">**🚨 HUMAN SCREAM DETECTED**</span>: High-confidence scream prediction. Triggers a visual banner and audio alert.

---

## 🛠️ Technical Details & Customization

### Dependencies
Key libraries used:
*   `librosa`: For all DSP and feature extraction tasks.
*   `torch_geometric`: For implementing the GGNN and graph data handling.
*   `scikit-learn`: For the SVM implementation.
*   `streamlit-webrtc`: For handling real-time browser-based audio streams.
*   `optuna`: For automated hyperparameter tuning.

### Extending the Framework
*   **New Features**: Edit `src/features.py` to add features like Chroma, Tonnetz, or completely custom embeddings.
*   **New Models**: Add new architectures in `src/models.py`. The `app.py` is designed to be modular—simply import your new model and add it to the ensemble logic.

---

## ⚠️ Disclaimer

This system is a **Proof of Concept (PoC)** designed for research and educational purposes. It is **not** a certified safety device and should not be relied upon for critical emergency response without rigorous validation and certification.
