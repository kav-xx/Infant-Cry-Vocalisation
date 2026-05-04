# 🍼 Infant Cry Emotion Classifier

A deep learning system that classifies infant cry emotions in real-time using entropy-based audio features and a hybrid CNN-BiLSTM model.

---

## 📌 Overview

Infants communicate distress through crying, but caregivers often struggle to distinguish the cause. This project tackles that problem by analyzing infant cry audio and predicting the underlying emotion using signal entropy features fed into a deep learning pipeline.

**Model Accuracy:** ~79.7% on 6-class classification  
**Interface:** Streamlit web app  
**Model Architecture:** CNN + Bidirectional LSTM

---

## 🎯 Emotion Classes

| Label | Description |
|-------|-------------|
| Belly Pain | Cry due to abdominal discomfort |
| Discomfort | General unease or irritation |
| Burping | Need to burp |
| Cold / Hot | Temperature-related distress |
| Hunger | Cry due to hunger |
| Tiredness | Fatigue or sleepiness |

---

## 🗂️ Project Structure

```
Infant-Cry-Classifier/
│
├── frontend.py                        # Streamlit web app
├── entropy_lstm_model.h5              # Trained LSTM model weights
│
├── cnn_bilstm_entropydtw_final.ipynb  # CNN-BiLSTM model training notebook
├── lstm_dtwentropy_final.ipynb        # LSTM with DTW+Entropy training notebook
├── preprocess4.ipynb                  # Audio preprocessing pipeline
├── preprocess4_dtw.ipynb              # DTW-based preprocessing pipeline
│
├── Results.docx                       # Evaluation results and analysis
└── README.md
```

---

## ⚙️ Model Architecture

The classification model uses a **hybrid CNN + Bidirectional LSTM** architecture:

```
Input (219 time frames × 12 entropy features)
        ↓
Conv1D (64 filters, kernel=7) → BatchNorm → MaxPool
        ↓
Conv1D (128 filters, kernel=5) → BatchNorm → MaxPool → Dropout(0.4)
        ↓
Bidirectional LSTM (128 units) → Dropout(0.4)
        ↓
Dense(64, ReLU) → Dense(6, Softmax)
```

**Training Config:**
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy (label smoothing=0.1)
- Split: 80% train / 10% validation / 10% test
- Cross-validation: 5-Fold Stratified

---

## 🔬 Feature Extraction

The system extracts 12 entropy-based features per audio clip:

| Feature | Description |
|---------|-------------|
| Shannon Entropy | Overall signal randomness |
| Spectral Entropy | Frequency-domain disorder |
| Multi-Scale Entropy (×3) | Entropy at 3 temporal scales |
| Bandwise Spectral Entropy (Low/Mid/High) | Per-band frequency entropy |
| Delta Entropy Mean | Rate of entropy change |
| Delta Entropy Std | Variability in entropy change |
| Signal Mean | Average amplitude |
| Signal Std Dev | Amplitude variability |

**Audio Preprocessing:**
- Sample Rate: 16,000 Hz
- Fixed Duration: 7 seconds (padded or trimmed)
- Normalization: Peak amplitude normalization

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/infant-cry-classifier.git
cd infant-cry-classifier
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` with:

```
streamlit
numpy
pandas
librosa
matplotlib
tensorflow
scipy
soundfile
scikit-learn
```

### Run the App

```bash
streamlit run frontend.py
```

Then open your browser at `http://localhost:8501`.

---

## 🖥️ App Usage

1. Launch the Streamlit app
2. Upload an infant cry audio file (`.wav`, `.mp3`, or `.ogg`)
3. View the **Waveform** and **Mel Spectrogram** visualizations
4. Inspect the **Spectral Entropy over Time** plot
5. Review the **12 extracted entropy features**
6. See the **predicted emotion** and confidence score

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~79.7% |
| Classes | 6 |
| Model | CNN-BiLSTM |
| Feature Type | Entropy-based |

Full evaluation metrics and confusion matrices are available in `Results.docx`.

---

## 🔮 Future Work

- Expand dataset for underrepresented classes
- Add real-time microphone recording support
- Explore transformer-based architectures
- Deploy as a mobile-friendly web application

---

## 👩‍💻 Authors

Developed as part of a Final Year Project (FYP) — Phase II.

---

## 📄 License

This project is intended for academic and research purposes.
