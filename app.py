import streamlit as st
import numpy as np
import os
import joblib
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import queue
import threading
from src import features, models

# Premium Tactical UI Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary-glow: #ff2e2e;
        --secondary-glow: #00ff9d;
        --bg-dark: #080a0f;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1f2e 0%, #080a0f 100%);
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
    }

    /* Glassmorphism Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.status-card) {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #fff 0%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Glow Status Cards */
    .status-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid transparent;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .status-normal {
        background: rgba(0, 255, 157, 0.05);
        border-color: rgba(0, 255, 157, 0.2);
        box-shadow: 0 0 20px rgba(0, 255, 157, 0.05);
        color: #00ff9d;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-alert {
        background: rgba(255, 46, 46, 0.1);
        border-color: rgba(255, 46, 46, 0.4);
        box-shadow: 0 0 40px rgba(255, 46, 46, 0.15);
        color: #ff2e2e;
        font-size: 28px;
        font-weight: 800;
        animation: threat-pulse 1.5s infinite;
    }

    @keyframes threat-pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 46, 46, 0.4); }
        70% { box-shadow: 0 0 0 20px rgba(255, 46, 46, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 46, 46, 0); }
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        color: #00ff9d !important;
        font-size: 42px !important;
    }

    /* Sidebar Fix */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.3) !important;
        border-right: 1px solid var(--glass-border);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
sensitivity = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.75, 0.05)
st.sidebar.markdown("---")
st.sidebar.write("System configured for tactical audio acquisition.")

@st.cache_resource
def load_models_demo():
    model_dir = 'scream_models'
    svm_path = os.path.join(model_dir, 'scream_svm.pkl')
    
    svm_model = None
    if os.path.exists(svm_path):
        svm_model = joblib.load(svm_path)
    
    ggnn_model = None
    config_path = os.path.join(model_dir, 'config.json')
    ggnn_path = os.path.join(model_dir, 'scream_ggnn.pt')
    
    if os.path.exists(config_path) and os.path.exists(ggnn_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            ggnn_model = models.ScreamGGNN(
                num_node_features=config['num_node_features'],
                hidden_channels=config['hidden_channels'],
                num_layers=config['num_layers']
            )
            ggnn_model.load_state_dict(torch.load(ggnn_path, map_location='cpu'))
            ggnn_model.eval()
        except:
            ggnn_model = None
    
    return svm_model, ggnn_model

svm_model, ggnn_model = load_models_demo()

# Thread-safe queue for audio data transmission between WebRTC thread and Streamlit Main thread
audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame):
        # Convert to numpy
        audio_data = frame.to_ndarray()
        
        # We only want the latest data for visualization/inference to reduce lag.
        try:
            audio_queue.put_nowait((audio_data, frame.sample_rate))
        except queue.Full:
            pass
            
        # Return None or empty to prevent loopback/playback in the browser
        return None

# INTERFACE
st.title("HUMAN SCREAM DETECTION")

# Tabs
tab_live, tab_upload, tab_metrics = st.tabs(["🔴 Live", "📂 Upload File", "📄 Details"])

with tab_live:
    col_main, col_stat = st.columns([2, 1])

    with col_main:
        webrtc_ctx = webrtc_streamer(
            key="scream-detection",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )
        
        # Spectrogram directly under the stream
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        fig_placeholder = st.empty()

    with col_stat:
        status_placeholder = st.empty()
        confidence_placeholder = st.empty()
        alert_placeholder = st.empty()

with tab_upload:
    st.subheader("Upload Audio for Safety Scan")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
    
    if uploaded_file is not None:
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file)
        
        # Analyze button
        if st.button("🚀 Analyze File"):
            with st.spinner("Analyzing structural and statistical patterns..."):
                target_sr = 22050
                
                # Split audio into 2-second windows with 50% overlap
                window_size = int(2.0 * target_sr)
                hop_size = int(1.0 * target_sr)
                
                # Generate Windows
                windows = []
                for i in range(0, len(y) - window_size + 1, hop_size):
                    windows.append(y[i:i+window_size])
                
                # If file is shorter than window_size
                if not windows:
                    windows = [y]
                
                max_final_prob = 0.0
                all_probs = []
                
                # Analyze each window
                for window in windows:
                    # Feature Extraction
                    feats = features.extract_features_from_array(window, sr=target_sr)
                    
                    svm_prob = 0.0
                    ggnn_prob = 0.0
                    
                    if svm_model:
                        # ENHANCED SVM VECTOR: Mean + Std of ALL 25 features for 50D input
                        m = np.mean(feats, axis=0)
                        s = np.std(feats, axis=0)
                        svm_vec = np.concatenate([m, s]).reshape(1, -1)
                        svm_prob = svm_model.predict_proba(svm_vec)[0][1]
                        
                    if ggnn_model:
                        try:
                            g_data = features.build_graph_from_features(feats)
                            from torch_geometric.data import Batch
                            batch = Batch.from_data_list([g_data])
                            with torch.no_grad():
                                out = ggnn_model(batch)
                                prob = torch.exp(out)
                                ggnn_prob = prob[0][1].item()
                        except:
                            pass
                    
                    # Ensemble for this window
                    if svm_model and ggnn_model:
                        current_prob = (svm_prob + ggnn_prob) / 2
                    else:
                        current_prob = max(svm_prob, ggnn_prob)
                    
                    all_probs.append(current_prob)
                
                # Final Decision: Max probability across windows
                final_prob = max(all_probs) if all_probs else 0.0
                
                # Results UI
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("Detection Confidence (Peak)", f"{final_prob:.2%}")
                    if final_prob > sensitivity:
                        st.error("🚨 HUMAN SCREAM DETECTED")
                    else:
                        st.success("🟢 STATUS: AMBIENT / SAFE")
                
                with res_col2:
                    # Static Spectrogram
                    fig, ax = plt.subplots(figsize=(8, 4))
                    plt.style.use('dark_background')
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
                    ax.set_title("Full Clip Spectral Analysis")
                    st.pyplot(fig)
                    plt.close(fig)

with tab_metrics:
    st.header("🧠 Neural Framework Architecture")
    
    # Architecture Diagram via Graphviz
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color="#e0e6ed", fontname="Outfit", fontcolor="black"];
        edge [color="gray"];
        
        subgraph cluster_0 {
            label = "Feature Extraction";
            style=dashed;
            MFCC [label="20 MFCCs"];
            Graph [label="Temporal Audio Graph"];
        }
        
        subgraph cluster_1 {
            label = "Neural Ensemble";
            style=dashed;
            SVM [label="SVM (RBF Kernel)"];
            GGNN [label="Gated Graph Neural Net"];
        }
        
        MFCC -> SVM;
        Graph -> GGNN;
        
        SVM -> Ensemble;
        GGNN -> Ensemble;
        
        Ensemble [label="Peak Search Ensemble", shape=diamond, color="#00ff9d"];
        Ensemble -> Output [label="Decision"];
        
        Output [label="Alert / Ambient", shape=oval, color="#ff2e2e"];
    }
    """)
    
    st.markdown("---")
    st.subheader("Model Performance")
    
    # Load config for display
    model_dir = 'scream_models'
    config_path = os.path.join(model_dir, 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)

    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.subheader("📈 Training Metrics")
        if config:
            st.write(f"**Total Dataset Size:** {config.get('total_samples', 3493)} Graphs")
            st.write(f"**Scream Samples:** {config.get('scream_samples', 862)}")
            st.write(f"**Ambient Samples:** {config.get('non_scream_samples', 2631)}")
            
            st.markdown("---")
            st.metric("SVM Accuracy", f"{config.get('svm_accuracy', 0.8112):.2%}")
            st.metric("GGNN (Gated Graph) Accuracy", f"{config.get('ggnn_accuracy', 0.8197):.2%}")
            st.caption("Accuracies shown are peak validation scores from Optuna Optimization.")
        else:
            st.info("Run training to populate detailed metrics.")

    with m_col2:
        st.subheader("🏗️ Ensemble Architecture")
        st.markdown("""
        **1. Statistical Branch (SVM)**
        - **Input**: 20-dimensional Global MFCC vector.
        - **Logic**: Uses a Radial Basis Function (RBF) kernel to find a hyperplane separating 'scream' frequency signatures from ambient noise.
        - **Role**: Captures the 'texture' of the sound.

        **2. Structural Branch (GGNN)**
        - **Type**: Gated Graph Neural Network.
        - **Message Passing**: Passes messages across temporal nodes (frames) for **%d layers**.
        - **Logic**: Models audio as a sequential graph where nodes represent 20ms frames and edges represent temporal flow.
        - **Internal**: Uses Gated Recurrent Units (GRU) to update node states.
        """ % config.get('num_layers', 4))

    st.markdown("---")
    st.subheader("🔍 Processing Logic: The Message Passing Scheme (MPS)")
    
    tech_col1, tech_col2 = st.columns([1, 2])
    
    with tech_col1:
        # Conceptual Graph Visualization
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.style.use('dark_background')
        # Simple line of nodes for "Audio Graph"
        nodes = np.arange(5)
        ax.scatter(nodes, [0]*5, s=200, color='#00FF00', zorder=5)
        for i in range(4):
            ax.annotate('', xy=(nodes[i+1], 0), xytext=(nodes[i], 0),
                        arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
        ax.set_title("Temporal Audio Graph (MPS)")
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        st.pyplot(fig)
        plt.close(fig)

    with tech_col2:
        st.info("**What is the MPS doing?**")
        st.write("""
        Traditional models look at frames in isolation. Our **Message Passing System (MPS)** ensures that the 'state' 
        of one audio frame is communicated to its neighbors. 
        
        Screams are not just high-frequency sounds; they have a specific **onset intensity** and **vocalic decay**. 
        By passing messages across **%d nodes**, the GGNN aggregates information over time, allowing the system 
        to distinguish between a short metallic clash (noise) and a sustained human shriek (scream).
        """ % config.get('num_layers', 4))
        
    st.subheader("⚙️ Auto-Tuning (Optuna)")
    st.write("""
    The models are not manually tuned. We use **Bayesian Optimization (Optuna)** to automatically search for:
    - **SVM**: Optimal 'C' (penalty) and 'Gamma' (influence).
    - **GGNN**: Hidden channel depth, number of message-passing layers, and learning rate.
    This ensures the framework adapts perfectly to whatever dataset it is trained on.
    """)

# Processing Loop
if webrtc_ctx.state.playing:
    # Buffer for audio to make a slightly longer window for spectrogram
    full_buffer = np.array([])
    buffer_duration_sec = 2.0
    
    while True:
        try:
            audio_chunk, sr = audio_queue.get(timeout=1.0)
            
            # Mix down to mono if stereo
            if audio_chunk.ndim > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Accumulate buffer
            # Resample if needed to 22050 for model
            # For efficiency, maybe resample only for model, keep sr for buffer?
            # Librosa load defaults to 22050. Let's enforce 22050.
            
            target_sr = 22050
            if sr != target_sr:
                # Fast resampling or use librosa (slow for realtime?)
                # For demo, simple decimation might work if ratio is int, else librosa
                audio_chunk_resampled = librosa.resample(audio_chunk.astype(float), orig_sr=sr, target_sr=target_sr)
            else:
                audio_chunk_resampled = audio_chunk.astype(float)
                
            # Run Inference on the chunk
            # We need a minimum length for features
            if len(audio_chunk_resampled) > 512:
                # Feature Extraction
                feats = features.extract_features_from_array(audio_chunk_resampled, sr=target_sr)
                
                # Check if energy is sufficient to be considered "Talking/Ambient"
                rms = np.sqrt(np.mean(audio_chunk_resampled**2))
                is_silent = rms < 0.005 # Noise gate
                
                scream_prob = 0.0
                
                if not is_silent:
                    # Models Inference
                    svm_prob = 0.0
                    ggnn_prob = 0.0
                    
                    # SVM
                    if svm_model:
                        try:
                            # ENHANCED SVM VECTOR: Mean + Std of ALL 25 features for 50D input
                            m = np.mean(feats, axis=0)
                            s = np.std(feats, axis=0)
                            svm_vec = np.concatenate([m, s]).reshape(1, -1)
                            svm_prob = svm_model.predict_proba(svm_vec)[0][1]
                        except:
                            pass
                            
                    # GGNN
                    if ggnn_model:
                        try:
                            g_data = features.build_graph_from_features(feats)
                            # Create batch (size 1)
                            from torch_geometric.data import Batch
                            batch = Batch.from_data_list([g_data])
                            
                            with torch.no_grad():
                                out = ggnn_model(batch)
                                prob = torch.exp(out) # Log softmax -> prob
                                ggnn_prob = prob[0][1].item()
                        except:
                            pass
                    
                    # Ensemble (averaged)
                    if svm_model and ggnn_model:
                        scream_prob = (svm_prob + ggnn_prob) / 2
                    elif svm_model:
                        scream_prob = svm_prob
                    elif ggnn_model:
                        scream_prob = ggnn_prob
                    else:
                        scream_prob = 0.0 # No models
                
                # UI Updates with Glow
                if scream_prob > sensitivity:
                    status_placeholder.markdown('<div class="status-card status-alert">THREAT DETECTED: HUMAN SCREAM</div>', unsafe_allow_html=True)
                    alert_placeholder.markdown("""
                        <div style='background: rgba(255, 46, 46, 0.05); padding: 15px; border-radius: 8px; border: 1px dashed #ff2e2e;'>
                            <p style='color:#ff2e2e; margin:0; font-weight:bold;'>⚠️ HIGH INTENSITY VOCAL EVENT DETECTED</p>
                            <p style='color:#ced4da; margin:0; font-size:12px;'>Source verification recommended.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if 'alert_active' not in st.session_state:
                        st.session_state['alert_active'] = True
                        import io
                        import scipy.io.wavfile
                        sample_rate, duration = 44100, 0.8
                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                        note = np.sin(2 * np.pi * 1000 * t) * (0.3 * 32767)
                        loud_note = note.astype(np.int16)
                        virtual_file = io.BytesIO()
                        scipy.io.wavfile.write(virtual_file, sample_rate, loud_note)
                        alert_placeholder.audio(virtual_file, format='audio/wav', autoplay=True)

                elif not is_silent:
                    status_placeholder.markdown('<div class="status-card status-normal">ACTIVE: ANALYZING SPEECH/AMBIENT</div>', unsafe_allow_html=True)
                    alert_placeholder.empty()
                    if 'alert_active' in st.session_state: del st.session_state['alert_active']
                else:
                    status_placeholder.markdown('<div class="status-card status-normal" style="opacity:0.6;">SYSTEM READY: MONITORING SILENCE</div>', unsafe_allow_html=True)
                    alert_placeholder.empty()
                    if 'alert_active' in st.session_state: del st.session_state['alert_active']

                confidence_placeholder.metric("Peak Confidence", f"{scream_prob:.2f}")

                # Improved High-Speed Spectrometer
                try:
                    plt.close('all') # Clear memory
                    fig, ax = plt.subplots(figsize=(10, 3), dpi=80)
                    fig.patch.set_facecolor('#080a0f')
                    ax.set_facecolor('#080a0f')
                    
                    # Log-frequency STFT
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_chunk_resampled, n_fft=512, hop_length=128)), ref=np.max)
                    
                    # Glow effect viz
                    img = librosa.display.specshow(D, sr=target_sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
                    ax.axis('off')
                    fig.tight_layout(pad=0)
                    fig_placeholder.pyplot(fig)
                except:
                    pass

        except queue.Empty:
            continue
        except Exception as e:
            # st.write(f"Global Loop Error: {e}")
            pass

st.markdown("---")
