# 🛡️ Safety Intelligence: Human Scream Detection Framework

A specialized security framework designed to detect human distress (screams) in real-time using a hybrid **Statistical (SVM)** and **Structural (Gated Graph Neural Network)** ensemble approach.

## 🚀 Key Features
*   **Real-Time Monitoring**: Low-latency detection using Streamlit and WebRTC.
*   **Hybrid Ensemble**: Combines the precision of Support Vector Machines with the temporal intelligence of Gated Graph Neural Networks.
*   **Peak Search**: Sliding window analysis for uploaded files to catch transient scream events.
*   **Auto-Optimization**: Fully automated hyperparameter tuning using **Optuna**.
*   **Safety Dashboard**: Interactive UI with spectral analysis and tactical alerting.

---

## 🔬 How It Works

### 1. Data Processing
Audio signals are transformed into multi-dimensional feature graphs:
*   **Nodes**: Each 20ms frame is a node containing 23 features (MFCCs, Zero Crossing Rate, Spectral Centroid, RMS Energy).
*   **Edges**: Temporal adjacency edges connect consecutive frames, forming a bidirectional graph.

### 2. The Models
*   **SVM (Support Vector Machine)**: Analyzes the global frequency distribution. It acts as a statistical filter to identify the broad "timbre" of a scream.
*   **GGNN (Gated Graph Neural Network)**: Processes the structural graph using a **Message Passing Scheme (MPS)**. It passes hidden states between nodes to capture the sustain and intensity patterns that distinguish human shrieks from mechanical noise.

### 3. Training & Auto-Tuning
The training script (`train.py`) handles the entire pipeline:
1.  **Acquisition**: Downloads the **Human Screaming Detection Dataset** (3493 total graphs).
2.  **Optuna Optimization**: Runs multiple trials to automatically find the best architecture (hidden channels, depth, LR for GGNN; C, Gamma for SVM).
3.  **Ensemble Saving**: Exports the optimized weights and configuration to `scream_models/`.

---

## 🛠️ Setup & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training (Auto-Hyperparameter Tuning)
Run the script to fetch data and optimize the models:
```bash
python train.py
```
*Note: This generates `scream_models/config.json` which the app uses to understand its own architecture.*

### 3. Launch Dashboard
```bash
streamlit run app.py
```
Access at: `http://localhost:8501`

---

## 📊 System Insights Tab
The Streamlit app includes a **System Deep-Dive** tab that displays:
*   **Metric Sliders**: Adjust sensitivity in real-time.
*   **Success Metrics**: View the peak accuracy achieved during training.
*   **MPS Visualization**: Understand how messages flow through the audio temporal graph.
*   **Architecture Logs**: Read the specific parameters discovered by Optuna.

---

## 📂 Project Structure
```text
├── src/
│   ├── data_loader.py    # Dataset acquisition & labeling
│   ├── features.py        # Graph construction & MFCC extraction
│   ├── models.py          # PyTorch GGNN & Sklearn SVM Pipelines
│   └── utils.py           # Model serialization
├── scream_models/         # Trained weights & Auto-tuned Config
├── app.py                 # Interactive Dashboard
├── train.py               # Auto-Optimization Script
├── requirements.txt
└── README.md
```

**Designed for Safety Intelligence & Tactical Audio Monitoring.** 🛡️
