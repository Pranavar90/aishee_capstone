# Safety Intelligence: Human Scream & Distress Detection System

A professional security framework that detects human screams and classifies the type of distress (fear, pain, anger) in real-time audio. The system uses advanced deep learning to distinguish between innocent high-pitched sounds (singing, baby cries) and actual crime-related distress screams.

---

## Understanding The System

### What Does This System Do?

Imagine you're building a smart security camera that doesn't just detect motion - it actually **listens** for human distress and can tell you whether someone is:
- Happy/excited (like at a party)
- In fear or pain (potential crime in progress)

The system works like this:

```
Microphone Audio → Computer Analysis → Alert Decision
     (Input)           (Brain)            (Output)
```

### The Two-Stage Detection Process

**Stage 1: Is it a scream?**
The first question is simple - "Is this a human scream or just background noise?" This is a binary decision (YES/NO).

**Stage 2: What type?**
If it IS a scream, the second stage analyzes what kind of emotion it contains:
- **Fear** - High priority alert (potential crime)
- **Pain** - High priority alert  
- **Anger** - Medium priority
- **Joy** - Low priority (false positive for crime detection)

This two-stage approach ensures we don't miss real emergencies while avoiding false alarms from concerts or happy crowds.

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT                              │
│                   (Microphone / File Upload)                    │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION                              │
│  • MFCCs (Frequency patterns)                                  │
│  • Jitter & Shimmer (Voice stability measures)                 │
│  • Spectral Centroid (Sound brightness)                        │
│  • Fundamental Frequency (Pitch)                               │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-TASK NEURAL NETWORK                       │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              SHARED ENCODER (GGNN + LSTM)               │   │
│   │         Analyzes temporal patterns in audio             │   │
│   └──────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
│         ┌───────────────┼───────────────┐                       │
│         ▼               ▼               ▼                       │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐                 │
│   │  HEAD A   │   │  HEAD B   │   │ DOMAIN    │                 │
│   │ Scream?   │   │ Emotion?  │   │ Adapter   │                 │
│   │ (YES/NO)  │   │ (Fear/    │   │ (Optional)│                 │
│   │           │   │  Pain/    │   │           │                 │
│   │           │   │  Joy/...) │   │           │                 │
│   └───────────┘   └───────────┘   └───────────┘                 │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION OUTPUT                              │
│                                                                 │
│   🎯 Scream Detected + Fear/Pain → RED ALERT (Crime Risk)      │
│   ✅ Scream Detected + Joy → YELLOW (Likely Not a Crime)        │
│   🟢 No Scream → GREEN (Safe/Ambient)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Component Explanation

#### 1. Feature Extraction (The "Ears")

When audio enters the system, we break it into small chunks (about 20 milliseconds each) and analyze what makes each chunk unique:

| Feature | What It Measures | Why It Helps |
|---------|-----------------|--------------|
| **MFCC** | Frequency patterns (like a musical fingerprint) | Identifies human voice vs other sounds |
| **Jitter** | How much the pitch wavers | Distress screams have more pitch instability |
| **Shimmer** | How much the volume wavers | Panic/fear = unsteady volume |
| **Spectral Centroid** | How "bright" or "sharp" the sound is | Screams are typically brighter |
| **F0 (Pitch)** | The fundamental pitch of the voice | High stress = distinctive pitch patterns |

#### 2. The Neural Network (The "Brain")

Instead of one simple model, we use a **Multi-Task Learning** approach with a shared backbone:

```
INPUT FEATURES (29 numbers per audio frame)
         │
         ▼
┌─────────────────┐
│  GGNN Layer 1   │ ← Processes each frame and its neighbors
│  GGNN Layer 2   │ ← Passes information across time
│  GGNN Layer 3   │ ← Builds understanding of patterns
│  GGNN Layer 4   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LSTM Layer      │ ← Understands how patterns evolve over time
└────────┬────────┘
         │
         ▼
    SHARED KNOWLEDGE (64 numbers representing the audio)
         │
    ┌────┴────┐
    ▼         ▼
HEAD A    HEAD B
(Scream)  (Emotion)
```

**Why Multi-Task?**
- Learning to detect screams ALSO helps learn about fear/anger
- Both tasks share the same underlying "understanding" of audio
- Makes the system faster (single model, not two)

#### 3. Domain Adaptation (The "Bridge")

The system can learn from different types of datasets:
- **Your main dataset**: Real scream recordings
- **Emotion datasets** (RAVDESS, CREMA-D): Acted emotions in studio

The Domain Adaptation component helps the model transfer knowledge between these different types of data, so it learns general patterns that work in the real world, not just in specific recording conditions.

---

## Key Capabilities

| Capability | Description |
|------------|-------------|
| **Real-Time Detection** | Process audio in under 1 second |
| **Emotion Classification** | Distinguish fear/anger from joy/excitement |
| **False Positive Filtering** | Won't alert on singing, baby cries, or crowds |
| **Configurable Sensitivity** | Adjust detection threshold via config file |
| **GPU Accelerated** | Uses CUDA for fast processing |

---

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `model_config.yaml` to adjust settings:

```yaml
detection:
  sensitivity_threshold: 50  # 0-100 (higher = more sensitive)
  
model:
  alpha: 0.6  # Balance between scream detection vs emotion classification
```

### Training the Model

The dataset is downloaded automatically (only once) and cached for future use.

```bash
# Train on CPU (recommended for most users)
python scripts/train_cpu.py
```

**What this does:**
1. Downloads the dataset (first time only - about 2.7GB)
2. Creates balanced training/validation sets
3. Trains the Multi-Head GGNN with DANN
4. Saves the model to `scream_models/`

**Training Options:**
```bash
# Default (CPU, 50 epochs, batch 32)
python scripts/train_cpu.py
```

The training uses:
- 85% of data for training, 15% for validation
- Early stopping if no improvement for 10 epochs
- Multi-task loss (60% scream detection + 30% emotion + 10% domain adaptation)

### Running the Application

```bash
# Run from the project root
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Using the App

1. **Upload Tab**: Upload an audio file (WAV, MP3, OGG)
2. **Click Analyze**: System processes the audio
3. **View Results**: See confidence score and alert status
4. **Details Tab**: Learn about the model architecture

---

## File Structure

```
aishee_capstone/
├── src/
│   ├── data_loader.py      # Dataset downloading & file listing
│   ├── features.py        # Audio feature extraction (MFCC, Jitter, etc.)
│   ├── models.py          # Neural network architectures (GGNN, Multi-Head)
│   └── utils.py           # Model saving/loading utilities
├── scripts/               # Training & Evaluation scripts
│   ├── train_cpu.py       # CPU training (recommended)
│   ├── train_final.py     # GPU training (faster)
│   ├── train_quick.py     # Fast prototype training
│   └── evaluate_prototype.py
├── scream_models/         # Trained model files
│   ├── multitask_ggnn.pt  # Model weights
│   ├── scaler.pkl         # Feature scaler
│   └── config.json        # Model configuration
├── model_config.yaml      # System settings (sensitivity, etc.)
├── app.py                 # Streamlit web interface
└── README.md              # This file
```

---

## Understanding the Results

When you analyze an audio file, you'll see:

| Result | Meaning |
|--------|---------|
| **Confidence > 50% + Fear/Pain** | High risk - likely a crime in progress |
| **Confidence > 50% + Joy** | Low risk - probably excited shouting (party, sports) |
| **Confidence < 50%** | No significant threat detected |

---

## Technical Details (For Developers)

- **Framework**: PyTorch + PyTorch Geometric (for Graph Neural Networks)
- **UI**: Streamlit
- **Audio Processing**: Librosa
- **Optimization**: Optuna (Bayesian hyperparameter tuning)
- **Model Type**: Gated Graph Neural Network (GGNN) + LSTM hybrid

The system achieves high accuracy by combining:
1. **GGNN** - Understands the temporal structure of audio
2. **LSTM** - Captures how patterns change over time
3. **Multi-Task Learning** - Shares knowledge between detection and classification
4. **Domain Adaptation** - Works across different recording environments

---

## Recent Updates (Multistage Training)
- **Architecture**: Implemented `MultiHeadGGNN` for enhanced feature extraction.
- **Organization**: Moved all training logic into the `/scripts` directory to clean up the root.
- **Models**: Included pre-trained weights for the Multi-Head GGNN and Scalers in `scream_models/`.
- **Efficiency**: Optimized data loading and feature extraction pipelines.

---

*This system is designed for safety and security applications. The accuracy metrics shown are based on training data and real-world performance may vary based on audio quality and environmental conditions.*