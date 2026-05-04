"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import entropy
from scipy.signal import stft

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Infant Cry Emotion Classifier",
    layout="wide"
)

st.title("🍼 Real-Time Infant Cry Emotion Classifier")

st.write(
    "Upload an infant cry audio file. The system extracts entropy-based features, "
    "visualizes waveform and spectrogram, and predicts the cry emotion using a trained LSTM model."
)

# -----------------------------
# Constants
# -----------------------------
SR = 16000
DURATION = 7
MAX_FRAMES = 219
NUM_FEATURES = 12

CLASS_LABELS = [
    "Belly Pain",
    "Discomfort",
    "Burping",
    "Cold / Hot",
    "Hunger",
    "Tiredness"
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("entropy_lstm_model.h5", compile=False)

model = load_model()

# -----------------------------
# Audio Preprocessing
# -----------------------------
def preprocess_audio(y):

    target_len = SR * DURATION

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    y = y / (np.max(np.abs(y)) + 1e-8)

    return y


# -----------------------------
# Entropy Feature Extraction
# -----------------------------
def extract_entropy_features(y):

    _, _, Zxx = stft(y, fs=SR, nperseg=512, noverlap=256)

    power = np.abs(Zxx) ** 2
    power /= np.sum(power, axis=0, keepdims=True) + 1e-10

    spectral_entropy = entropy(power, axis=0)

    shannon_entropy = []

    frame_len = 512

    for i in range(0, len(y) - frame_len, frame_len):

        frame = y[i:i + frame_len]

        hist, _ = np.histogram(frame, bins=50, density=True)

        hist += 1e-10

        shannon_entropy.append(entropy(hist))

    shannon_entropy = np.array(shannon_entropy)

    min_len = min(len(spectral_entropy), len(shannon_entropy))

    spectral_entropy = spectral_entropy[:min_len]
    shannon_entropy = shannon_entropy[:min_len]

    features = np.stack([shannon_entropy, spectral_entropy], axis=1)

    if features.shape[0] < MAX_FRAMES:

        pad = MAX_FRAMES - features.shape[0]

        features = np.pad(features, ((0, pad), (0, 0)))

    else:

        features = features[:MAX_FRAMES]

    features = np.pad(features, ((0, 0), (0, NUM_FEATURES - features.shape[1])))

    features = (features - features.mean()) / (features.std() + 1e-8)

    return features


# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3", "ogg"]
)
st.audio(uploaded_file, format="audio/wav")
if uploaded_file is not None:

    import soundfile as sf

    y, sr = sf.read(uploaded_file)
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = preprocess_audio(y)

    # =============================
    # AUDIO VISUALIZATION
    # =============================
    st.subheader("🎵 Audio Visualization")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots()

        librosa.display.waveshow(y, sr=SR, ax=ax)

        ax.set_title("Waveform")

        st.pyplot(fig)

    with col2:

        mel = librosa.feature.melspectrogram(y=y, sr=SR)

        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots()

        img = librosa.display.specshow(
            mel_db,
            sr=SR,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )

        ax.set_title("Mel Spectrogram")

        fig.colorbar(img, ax=ax)

        st.pyplot(fig)

    # =============================
    # FEATURE EXTRACTION
    # =============================
    features = extract_entropy_features(y)

    feature_values = features.mean(axis=0)

    feature_names = [
        "Shannon Entropy",
        "Spectral Entropy",
        "Feature 3",
        "Feature 4",
        "Feature 5",
        "Feature 6",
        "Feature 7",
        "Feature 8",
        "Feature 9",
        "Feature 10",
        "Feature 11",
        "Feature 12",
    ]

    df = pd.DataFrame({
        "Feature Name": feature_names,
        "Value": feature_values
    })

    st.subheader("📊 Extracted Entropy Features")

    col3, col4 = st.columns(2)

    with col3:

        st.dataframe(df, use_container_width=True)

    with col4:

        fig, ax = plt.subplots()

        ax.bar(feature_names, feature_values)

        ax.set_xticklabels(feature_names, rotation=60)

        ax.set_title("Feature Bar Chart")

        st.pyplot(fig)

    # =============================
    # PREDICTION
    # =============================
    features_input = np.expand_dims(features, axis=0)

    probs = model.predict(features_input, verbose=0)[0]

    pred_idx = np.argmax(probs)

    emotion = CLASS_LABELS[pred_idx]

    confidence = probs[pred_idx] * 100

    st.markdown("---")

    st.success(f"🍼 Predicted Emotion: **{emotion}**")

    st.progress(int(confidence))

    st.write(f"Confidence: {confidence:.2f}%")
"""
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Infant Cry Emotion Classifier", layout="wide")

st.title("🍼 Real-Time Infant Cry Emotion Classifier")

st.write(
    "Upload an infant cry audio file. The system extracts entropy-based features, "
    "visualizes waveform and spectrogram, and predicts the emotion using a trained LSTM model."
)
st.markdown("""
### System Pipeline
1️⃣ Audio Upload  
2️⃣ Audio Normalization  
3️⃣ Entropy Feature Extraction  
4️⃣ LSTM Classification  
5️⃣ Emotion Prediction
""")

st.info("Model: Entropy-based LSTM | Test Accuracy: 79.7% | Classes: 6 Infant Cry Emotions")

# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
SR = 16000
DURATION = 7
FIXED_TIME_FRAMES = 219

CLASS_LABELS = [
    "Belly Pain",
    "Discomfort",
    "Burping",
    "Cold / Hot",
    "Hunger",
    "Tiredness"
]

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("entropy_lstm_model.h5", compile=False)

model = load_model()

# -------------------------------------------------------
# ENTROPY FUNCTIONS
# -------------------------------------------------------
def shannon_entropy(signal, bins=100):

    hist, _ = np.histogram(signal, bins=bins)

    prob = hist / np.sum(hist)

    prob = prob[prob > 0]

    return -np.sum(prob * np.log2(prob))


def spectral_entropy(y):

    S = np.abs(librosa.stft(y)) ** 2

    psd = S / np.sum(S, axis=0, keepdims=True)

    ent = -np.sum(psd * np.log2(psd + 1e-12), axis=0)

    return np.nan_to_num(ent)


def multi_scale_entropy(signal, scales=3):

    entropies = []

    for scale in range(1, scales + 1):

        n = len(signal) // scale

        coarse = np.mean(signal[:n * scale].reshape(n, scale), axis=1)

        entropies.append(shannon_entropy(coarse))

    return np.array(entropies)


def bandwise_entropy(y):

    S = np.abs(librosa.stft(y)) ** 2

    freqs = librosa.fft_frequencies(sr=SR)

    band_edges = np.linspace(0, SR / 2, 4)

    band_entropy = []

    for i in range(3):

        idx = np.where((freqs >= band_edges[i]) & (freqs < band_edges[i + 1]))[0]

        band = S[idx, :]

        psd = band / (np.sum(band, axis=0, keepdims=True) + 1e-12)

        ent = -np.sum(psd * np.log2(psd + 1e-12), axis=0)

        band_entropy.append(np.mean(ent))

    return np.array(band_entropy)

# -------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------
def extract_features(y):

    frame_entropy = spectral_entropy(y)

    delta_entropy = np.diff(frame_entropy, prepend=frame_entropy[0])

    shannon_val = shannon_entropy(y)

    mse = multi_scale_entropy(y)

    band_vals = bandwise_entropy(y)

    mean_signal = np.mean(y)
    std_signal = np.std(y)

    spectral_mean = np.mean(frame_entropy)

    delta_mean = np.mean(np.abs(delta_entropy))
    delta_std = np.std(delta_entropy)

    feature_vector = np.array([
        shannon_val,
        spectral_mean,
        mse[0],
        mse[1],
        mse[2],
        band_vals[0],
        band_vals[1],
        band_vals[2],
        delta_mean,
        delta_std,
        abs(mean_signal),
        std_signal
    ])

    feature_vector = np.clip(feature_vector, 0, None)

    features_seq = np.tile(feature_vector, (FIXED_TIME_FRAMES, 1))

    features_seq = np.expand_dims(features_seq, axis=0)

    return feature_vector, features_seq, frame_entropy


feature_names = [
    "Shannon Entropy",
    "Spectral Entropy",
    "Multi-Scale Entropy (1)",
    "Multi-Scale Entropy (2)",
    "Multi-Scale Entropy (3)",
    "Bandwise Spectral Entropy (Low)",
    "Bandwise Spectral Entropy (Mid)",
    "Bandwise Spectral Entropy (High)",
    "Delta Entropy Mean",
    "Delta Entropy Std",
    "Mean of the Signal",
    "Standard Deviation of the  Signal"
]

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
uploaded_file = st.file_uploader("Upload Infant Cry Audio", type=["wav","mp3","ogg"])
st.audio(uploaded_file, format="audio/wav")
if uploaded_file is not None:

    y,_ = librosa.load(uploaded_file, sr=SR)

    target_len = SR * DURATION

    if len(y) < target_len:
        y = np.pad(y,(0,target_len-len(y)))
    else:
        y = y[:target_len]

    y = y / (np.max(np.abs(y)) + 1e-8)

    # -------------------------------------------------------
    # AUDIO VISUALIZATION
    # -------------------------------------------------------
    st.subheader("🎧 Audio Visualization")

    col1,col2 = st.columns(2)

    with col1:

        fig,ax = plt.subplots()

        librosa.display.waveshow(y,sr=SR,ax=ax)

        ax.set_title("Waveform")

        st.pyplot(fig)

    with col2:

        mel = librosa.feature.melspectrogram(y=y,sr=SR)

        mel_db = librosa.power_to_db(mel,ref=np.max)

        fig,ax = plt.subplots()

        img = librosa.display.specshow(
            mel_db,
            sr=SR,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )

        ax.set_title("Mel Spectrogram")

        fig.colorbar(img,ax=ax)

        st.pyplot(fig)

    # -------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------
    feature_vector,features_seq,frame_entropy = extract_features(y)

    # -------------------------------------------------------
    # ENTROPY OVER TIME PLOT (NEW)
    # -------------------------------------------------------
    st.subheader("📈 Spectral Entropy Over Time")

    fig, ax = plt.subplots(figsize=(5,2))   # smaller plot

    ax.plot(frame_entropy, color="purple", linewidth=1)

    ax.set_xlabel("Frame", fontsize=8)

    ax.set_ylabel("Entropy", fontsize=8)

    ax.set_title("Temporal Entropy Variation", fontsize=10)

    plt.tight_layout()

    st.pyplot(fig)

    # -------------------------------------------------------
    # FEATURE TABLE + BAR CHART
    # -------------------------------------------------------
    st.subheader("📊 Extracted Entropy Features")

    df = pd.DataFrame({
        "Feature Name":feature_names,
        "Value":feature_vector
    })

    col3,col4 = st.columns(2)

    with col3:

        st.dataframe(df,use_container_width=True)

    with col4:

        fig,ax = plt.subplots()

        ax.barh(feature_names,feature_vector)

        ax.set_title("Feature Bar Chart")

        st.pyplot(fig)

    # -------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------
    probs = model.predict(features_seq,verbose=0)[0]

    pred_idx = np.argmax(probs)

    emotion = CLASS_LABELS[pred_idx]

    confidence = probs[pred_idx] * 100

    st.markdown("---")

    st.success(f"🍼 Predicted Emotion: **{emotion}**")

    st.progress(int(confidence))

    st.write(f"Confidence: {confidence:.2f}%")
