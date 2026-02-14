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

# Page Config
st.set_page_config(
    page_title="Safety Intelligence Framework",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Safety Aesthetic"
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }
    .status-normal {
        color: #00FF00;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 255, 0, 0.1);
        text-align: center;
    }
    .status-warning {
        color: #FFA500;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 165, 0, 0.1);
        text-align: center;
    }
    .status-alert {
        color: #FF0000;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 0, 0, 0.2);
        text-align: center;
        font-size: 24px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
sensitivity = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.75, 0.05)
st.sidebar.markdown("---")
st.sidebar.info("Model Status")

# Load Models
@st.cache_resource
def load_models_demo():
    model_dir = 'scream_models' # Adjust if needed
    svm_path = os.path.join(model_dir, 'scream_svm.pkl')
    # Use saved torch model path, or initialize class if needed
    # For this demo, we need to know the hidden channels and layers used during training.
    # We'll try to load them safely or use defaults.
    
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
        except Exception as e:
            st.sidebar.error(f"Error loading GGNN: {e}")
            ggnn_model = None
    
    return svm_model, ggnn_model

svm_model, ggnn_model = load_models_demo()

if svm_model:
    st.sidebar.success("SVM Loaded ✅")
else:
    st.sidebar.warning("SVM Not Found ❌ (Run training first)")

if ggnn_model:
    st.sidebar.success("GGNN Loaded ✅")
else:
    st.sidebar.warning("GGNN Not Found ❌ (Run training first)")

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

st.title("🛡️ Safety Intelligence: Scream Detection")
st.markdown("Real-time audio monitoring system using **SVM + GGNN Ensemble**.")

# Tabs for Mode Selection
tab_live, tab_upload = st.tabs(["🔴 Live Monitoring", "📂 File Analysis"])

with tab_live:
    # Two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Monitor")
        
        webrtc_ctx = webrtc_streamer(
            key="scream-detection",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
        )

    with col2:
        st.subheader("System Status")
        status_placeholder = st.empty()
        alert_placeholder = st.empty()
        confidence_placeholder = st.empty()

    # Spectrogram Placeholder
    st.subheader("Spectral Analysis")
    fig_placeholder = st.empty()

with tab_upload:
    st.subheader("Upload Audio for Safety Scan")
    uploaded_file = st.file_uploader("Choose a WAV/MP3 file", type=["wav", "mp3"])
    
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
                        mfccs = feats[:, :20]
                        global_mfccs = np.mean(mfccs, axis=0).reshape(1, -1)
                        svm_prob = svm_model.predict_proba(global_mfccs)[0][1]
                        
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
                            # SVM expects global average of first 20 MFCCs
                            mfccs = feats[:, :20]
                            global_mfccs = np.mean(mfccs, axis=0).reshape(1, -1)
                            svm_prob = svm_model.predict_proba(global_mfccs)[0][1]
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
                
                # Update UI
                if scream_prob > sensitivity:
                    status_placeholder.markdown('<div class="status-alert">🚨 ALERT: SCREAM DETECTED</div>', unsafe_allow_html=True)
                    alert_placeholder.markdown("### ⚠️ HIGH PRIORITY")
                    
                    # Sound Alert
                    # Generate a beep (sine wave)
                    # To avoid re-generating every frame, we could cache it, but it's fast.
                    # Or check if we already played it recently to avoid stutter.
                    # For a demo, let's just show the audio player which tries to autoplay.
                    
                    # 1kHz sine wave for 0.5s
                    # We can't easily push audio back through WebRTC in this mode (SendRecv usually for video/audio transform).
                    # We'll use st.audio which might refresh the app structure, which is not ideal in a loop.
                    # A better way is to assume the visual alert is sufficient for the "demo" constraint 
                    # OR use a placeholder that we update.
                    
                    # NOTE: Updating st.audio in a loop might cause the frontend to re-mount the player.
                    # We will skip the actual audio element to maintain stability, as requested "trigger ... sound alert" 
                    # is best done via client-side JS, which is hard here.
                    # But we can try a single placeholder update if not already alerting.
                    
                    # simple beep logic:
                    if 'alert_active' not in st.session_state:
                        st.session_state['alert_active'] = True
                        # Play sound
                        # Create a dummy beep
                        import io
                        import scipy.io.wavfile
                        
                        sample_rate = 44100
                        duration = 1.0
                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                        note = np.sin(2 * np.pi * 1000 * t) * (0.5 * 32767)
                        loud_note = note.astype(np.int16)
                        
                        virtual_file = io.BytesIO()
                        scipy.io.wavfile.write(virtual_file, sample_rate, loud_note)
                        
                        # st.audio(virtual_file, format='audio/wav', autoplay=True) # Autoplay might need user interaction first
                        # We'll just display it.
                        alert_placeholder.audio(virtual_file, format='audio/wav', autoplay=True)

                elif not is_silent:
                    status_placeholder.markdown('<div class="status-warning">🗣️ Talking / Ambient</div>', unsafe_allow_html=True)
                    alert_placeholder.empty()
                    if 'alert_active' in st.session_state:
                        del st.session_state['alert_active']
                else:
                    status_placeholder.markdown('<div class="status-normal">🟢 Monitoring (Silence)</div>', unsafe_allow_html=True)
                    alert_placeholder.empty()
                    if 'alert_active' in st.session_state:
                         del st.session_state['alert_active']

                confidence_placeholder.metric("Scream Probability", f"{scream_prob:.2f}")

                confidence_placeholder.metric("Scream Probability", f"{scream_prob:.2f}")

                # Update Spectrogram
                try:
                    # Create or update plot
                    # Using a simpler plot if fig_placeholder has issues
                    fig, ax = plt.subplots(figsize=(8, 3))
                    plt.style.use('dark_background')
                    
                    # Compute STFT
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_chunk_resampled, n_fft=512)), ref=np.max)
                    
                    # Display
                    librosa.display.specshow(D, sr=target_sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
                    ax.set_title("Live Spectrometer", color='white', fontsize=10)
                    ax.set_xlabel("Time (s)", color='gray', fontsize=8)
                    ax.set_ylabel("Freq (Hz)", color='gray', fontsize=8)
                    
                    # Use a fixed layout to prevent jumping
                    fig.tight_layout()
                    
                    # Render to placeholder
                    fig_placeholder.pyplot(fig)
                    plt.close(fig)
                except Exception as vis_error:
                    # st.write(f"Vis Error: {vis_error}")
                    pass

        except queue.Empty:
            continue
        except Exception as e:
            # st.write(f"Global Loop Error: {e}")
            pass

st.markdown("---")
st.caption("Safety Intelligence System - Modular Framework PoC")
