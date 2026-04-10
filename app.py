import streamlit as st
import numpy as np
import os
import joblib
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import yaml
from src import features, models
from sklearn.preprocessing import StandardScaler


def load_config():
    config_path = os.path.join(os.getcwd(), "model_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "detection": {"sensitivity_threshold": 50},
        "model": {"alpha": 0.6, "hidden_channels": 64, "num_layers": 4},
        "training": {"epochs": 50, "batch_size": 32},
    }


config = load_config()


def get_sensitivity_threshold():
    return config.get("detection", {}).get("sensitivity_threshold", 50) / 100.0


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --accent-cyan: #00d4ff;
        --accent-red: #ff3366;
        --accent-green: #00ff88;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --text-muted: rgba(255, 255, 255, 0.4);
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
    }

    /* Ambient Background Effect */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(0, 212, 255, 0.03) 0%, transparent 50%),
                    radial-gradient(circle at 70% 80%, rgba(255, 51, 102, 0.03) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    .stMainBlockContainer {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Glass Panel Base */
    .glass-panel {
        background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 28px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
    }

    .glass-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }

    /* Upload Zone */
    .upload-zone {
        border: 2px dashed rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 4rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.02);
    }

    .upload-zone:hover {
        border-color: rgba(0, 212, 255, 0.3);
        background: rgba(0, 212, 255, 0.02);
    }

    .upload-icon {
        width: 80px;
        height: 80px;
        margin: 0 auto 1.5rem;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,212,255,0.05));
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }

    .upload-text {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
    }

    .upload-hint {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }

    /* Status Cards */
    .status-card {
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .status-secure {
        background: linear-gradient(135deg, rgba(0,255,136,0.08) 0%, rgba(0,255,136,0.02) 100%);
        border: 1px solid rgba(0,255,136,0.2);
        animation: glow-green 3s ease-in-out infinite;
    }

    .status-alert {
        background: linear-gradient(135deg, rgba(255,51,102,0.1) 0%, rgba(255,51,102,0.02) 100%);
        border: 1px solid rgba(255,51,102,0.3);
        animation: glow-red 2s ease-in-out infinite;
    }

    @keyframes glow-green {
        0%, 100% { box-shadow: 0 0 30px rgba(0,255,136,0.1), inset 0 0 30px rgba(0,255,136,0.02); }
        50% { box-shadow: 0 0 60px rgba(0,255,136,0.2), inset 0 0 60px rgba(0,255,136,0.05); }
    }

    @keyframes glow-red {
        0%, 100% { box-shadow: 0 0 30px rgba(255,51,102,0.15), inset 0 0 30px rgba(255,51,102,0.03); }
        50% { box-shadow: 0 0 60px rgba(255,51,102,0.3), inset 0 0 60px rgba(255,51,102,0.08); }
    }

    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.75rem;
        animation: pulse-dot 2s ease-in-out infinite;
    }

    .status-secure .status-indicator {
        background: var(--accent-green);
        box-shadow: 0 0 20px var(--accent-green);
    }

    .status-alert .status-indicator {
        background: var(--accent-red);
        box-shadow: 0 0 20px var(--accent-red);
        animation-duration: 1s;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    .confidence-big {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4.5rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 1.5rem 0;
    }

    .status-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
    }

    .status-text {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 0.05em;
    }

    .status-secure .status-text { color: var(--accent-green); }
    .status-alert .status-text { color: var(--accent-red); }

    /* Emotion Spectrum */
    .emotion-section {
        margin-top: 2rem;
    }

    .emotion-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 1.25rem;
    }

    .emotion-row {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }

    .emotion-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        width: 80px;
        flex-shrink: 0;
    }

    .emotion-bar-track {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        overflow: hidden;
        margin: 0 1rem;
    }

    .emotion-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .emotion-bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    .emotion-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-primary);
        width: 50px;
        text-align: right;
        flex-shrink: 0;
    }

    .emotion-neutral .emotion-bar-fill { background: linear-gradient(90deg, #3498db, #5dade2); box-shadow: 0 0 15px rgba(52,152,219,0.5); }
    .emotion-happy .emotion-bar-fill { background: linear-gradient(90deg, #f1c40f, #f4d03f); box-shadow: 0 0 15px rgba(241,196,15,0.5); }
    .emotion-sad .emotion-bar-fill { background: linear-gradient(90deg, #3498db, #85c1e9); box-shadow: 0 0 15px rgba(52,152,219,0.5); }
    .emotion-angry .emotion-bar-fill { background: linear-gradient(90deg, #e67e22, #f39c12); box-shadow: 0 0 15px rgba(230,126,34,0.5); }
    .emotion-fearful .emotion-bar-fill { background: linear-gradient(90deg, #9b59b6, #bb8fce); box-shadow: 0 0 15px rgba(155,89,182,0.5); }

    /* Spectrogram */
    .spectro-container {
        background: rgba(0,0,0,0.3);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        padding: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        padding: 1rem 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-secondary);
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
        border-bottom-color: var(--accent-cyan) !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(0,212,255,0.05) 100%);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 16px;
        color: var(--accent-cyan);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 1rem 3rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,212,255,0.25) 0%, rgba(0,212,255,0.1) 100%);
        border-color: var(--accent-cyan);
        box-shadow: 0 0 40px rgba(0,212,255,0.2);
        transform: translateY(-2px);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: var(--accent-cyan) !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
    }

    /* Details Section */
    .details-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }

    .details-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-secondary);
    }

    .details-text strong {
        color: var(--text-primary);
        font-weight: 600;
    }

    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }

    .config-item {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .config-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .config-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 2rem 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.2);
    }

    /* File uploader styling */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.02) !important;
        border: 2px dashed rgba(255,255,255,0.1) !important;
        border-radius: 24px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(0,212,255,0.3) !important;
        background: rgba(0,212,255,0.02) !important;
    }

    [data-testid="stFileUploaderFileName"] {
        background: rgba(0,212,255,0.1) !important;
        border: 1px solid rgba(0,212,255,0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "scream_models"
    svm_path = os.path.join(model_dir, "scream_svm.pkl")

    svm_model = None
    if os.path.exists(svm_path):
        svm_model = joblib.load(svm_path)

    ggnn_model = None
    config_path = os.path.join(model_dir, "config.json")
    ggnn_path = os.path.join(model_dir, "scream_ggnn.pt")

    if os.path.exists(config_path) and os.path.exists(ggnn_path):
        try:
            import json

            with open(config_path, "r") as f:
                model_config = json.load(f)

            num_emotion = model_config.get("num_emotion_classes", 5)
            hidden_ch = model_config.get("hidden_channels", 128)
            lstm_hid = model_config.get("lstm_hidden", 128)

            if model_config.get("model_type") == "MultiHeadGGNN":
                ggnn_model = models.MultiHeadGGNN(
                    num_node_features=model_config["num_node_features"],
                    hidden_channels=hidden_ch,
                    num_layers=model_config["num_layers"],
                    num_emotion_classes=num_emotion,
                    lstm_hidden=lstm_hid,
                    use_dann=model_config.get("use_dann", True),
                )
            else:
                ggnn_model = models.ScreamGGNN(
                    num_node_features=model_config["num_node_features"],
                    hidden_channels=hidden_ch,
                    num_layers=model_config["num_layers"],
                )

            ggnn_model.load_state_dict(
                torch.load(ggnn_path, map_location=device, weights_only=True)
            )
            ggnn_model.to(device)
            ggnn_model.eval()
        except Exception as e:
            st.warning(f"Error loading GGNN model: {e}")
            ggnn_model = None

    return svm_model, ggnn_model, device


svm_model, ggnn_model, device = load_models_demo()

model_meta = {}
meta_path = os.path.join("scream_models", "config.json")
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        import json

        model_meta = json.load(f)

st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

tab_upload, tab_metrics = st.tabs(["Analyze", "System"])

with tab_upload:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop audio file or click to upload",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file)

        if st.button("Run Analysis"):
            with st.spinner("Processing audio..."):
                target_sr = 22050

                window_size = int(2.0 * target_sr)
                hop_size = int(1.0 * target_sr)

                windows = []
                for i in range(0, len(y) - window_size + 1, hop_size):
                    windows.append(y[i : i + window_size])

                if not windows:
                    windows = [y]

                all_probs = []
                current_emotion_probs = None

                for window in windows:
                    feats_svm = features.extract_features_from_array(
                        window, sr=target_sr, include_stress=False
                    )
                    feats_ggnn = features.extract_features_from_array(
                        window, sr=target_sr, include_stress=True
                    )

                    svm_prob = 0.0
                    ggnn_prob = 0.0

                    if svm_model:
                        m = np.mean(feats_svm, axis=0)
                        s = np.std(feats_svm, axis=0)
                        svm_vec = np.concatenate([m, s]).reshape(1, -1)
                        svm_prob = svm_model.predict_proba(svm_vec)[0][1]

                    if ggnn_model:
                        try:
                            g_data = features.build_graph_from_features(feats_ggnn)
                            from torch_geometric.data import Batch

                            batch = Batch.from_data_list([g_data]).to(device)
                            with torch.no_grad():
                                out = ggnn_model(batch)

                                if isinstance(out, (list, tuple)):
                                    scream_logits = out[0]
                                    emotion_logits = out[1]

                                    s_prob = torch.exp(scream_logits)
                                    e_prob = torch.exp(emotion_logits)

                                    ggnn_prob = s_prob[0][1].item()
                                    current_emotion_probs = e_prob[0].cpu().numpy()
                                else:
                                    prob = torch.exp(out)
                                    ggnn_prob = prob[0][1].item()
                                    current_emotion_probs = None
                        except Exception as e:
                            pass

                    if svm_model and ggnn_model:
                        current_prob = (svm_prob + ggnn_prob) / 2
                    else:
                        current_prob = max(svm_prob, ggnn_prob)

                    all_probs.append(current_prob)

                final_prob = max(all_probs) if all_probs else 0.0
                emotion_labels = model_meta.get(
                    "emotions", ["Neutral", "Happy", "Sad", "Angry", "Fearful"]
                )
                is_alert = final_prob > get_sensitivity_threshold()

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None and "final_prob" in dir():
        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            status_class = "status-alert" if is_alert else "status-secure"
            status_text = "Scream Detected" if is_alert else "System Secure"

            st.markdown(
                f"""
            <div class="status-card {status_class}">
                <div class="status-label">Detection Confidence</div>
                <div class="confidence-big" style="color: {"#ff3366" if is_alert else "#00ff88"};">{final_prob:.1%}</div>
                <div class="status-text"><span class="status-indicator"></span>{status_text}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            if current_emotion_probs is not None and len(current_emotion_probs) > 0:
                st.markdown('<div class="emotion-section">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="emotion-title">Emotional Signature</div>',
                    unsafe_allow_html=True,
                )

                emotion_map = {
                    "Neutral": "neutral",
                    "Happy": "happy",
                    "Sad": "sad",
                    "Angry": "angry",
                    "Fearful": "fearful",
                }

                for i, label in enumerate(emotion_labels):
                    if i < len(current_emotion_probs):
                        val = current_emotion_probs[i]
                        css_class = emotion_map.get(label, "neutral")
                        bar_width = float(val) * 100

                        st.markdown(
                            f"""
                        <div class="emotion-row">
                            <div class="emotion-name">{label}</div>
                            <div class="emotion-bar-track">
                                <div class="emotion-bar-fill {css_class}" style="width: {bar_width:.1f}%;"></div>
                            </div>
                            <div class="emotion-value">{val:.1%}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="spectro-container">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.style.use("dark_background")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(
                D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="magma"
            )
            ax.set_title(
                "Frequency Spectrum",
                color="white",
                pad=15,
                fontsize=11,
                fontfamily="JetBrains Mono",
            )
            fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.02)
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

with tab_metrics:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="details-header">Neural Architecture</div>', unsafe_allow_html=True
    )

    st.markdown(
        """
    <div class="details-text">
        The system utilizes a <strong>Gated Graph Neural Network</strong> with specialized output heads for multi-task learning.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    if model_meta:
        with col1:
            st.metric("Accuracy", f"{model_meta.get('scream_accuracy', 0):.1%}")
        with col2:
            st.metric("Samples", model_meta.get("total_samples", "N/A"))
        with col3:
            st.metric("Layers", model_meta.get("num_layers", "N/A"))

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="details-header" style="font-size: 1.1rem;">Components</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="details-text">
        <strong>1. Structural Backbone</strong><br>
        Gated Graph Neural Network with message passing across temporal nodes for contextual audio analysis.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="details-text" style="margin-top: 1rem;">
        <strong>2. Temporal Memory</strong><br>
        LSTM layer captures long-range dependencies, distinguishing sustained vocal patterns from transient noise.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="details-text" style="margin-top: 1rem;">
        <strong>3. Multi-Head Output</strong><br>
        Detection Head (scream vs ambient) + Emotion Head + Domain Adaptation via DANN.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="details-header" style="font-size: 1.1rem;">Configuration</div>',
        unsafe_allow_html=True,
    )

    if model_meta:
        config_cols = st.columns(3)
        configs = [
            ("Hidden Channels", model_meta.get("hidden_channels", "N/A")),
            ("LSTM Hidden", model_meta.get("lstm_hidden", "N/A")),
            ("Emotion Classes", model_meta.get("num_emotion_classes", "N/A")),
            ("Node Features", model_meta.get("num_node_features", "N/A")),
            ("Domain Adapt", "Yes" if model_meta.get("use_dann", False) else "No"),
            ("Model Type", model_meta.get("model_type", "N/A")),
        ]

        for i, (label, value) in enumerate(configs):
            with config_cols[i % 3]:
                st.markdown(
                    f"""
                <div class="config-item">
                    <div class="config-label">{label}</div>
                    <div class="config-value">{value}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)
