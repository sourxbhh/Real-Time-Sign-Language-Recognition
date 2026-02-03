# Real-Time Sign Language Recognition and Speech Translation System

A production-ready end-to-end system that captures live video, recognizes ASL gestures,
translates them to English text, and converts to spoken audio.

## 🏗️ Project Structure

```
SignLang/
├── src/
│   ├── __init__.py
│   ├── webcam/                 # Video capture module
│   │   ├── __init__.py
│   │   └── capture.py
│   ├── detection/              # Hand/pose landmark detection
│   │   ├── __init__.py
│   │   └── landmark_detector.py
│   ├── recognition/            # Gesture recognition model
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── inference.py
│   ├── translation/            # Sign to text translation
│   │   ├── __init__.py
│   │   └── translator.py
│   ├── speech/                 # Text-to-speech engine
│   │   ├── __init__.py
│   │   └── tts_engine.py
│   └── utils/                  # Utilities and helpers
│       ├── __init__.py
│       ├── buffer.py
│       └── preprocessing.py
├── training/                   # Model training pipeline
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   └── augmentation.py
├── ui/                         # User interface
│   ├── __init__.py
│   ├── app.py                  # Streamlit app
│   └── components.py
├── models/                     # Saved model weights
│   └── .gitkeep
├── data/                       # Training data
│   ├── raw/
│   └── processed/
├── config/
│   └── config.yaml
├── tests/
│   └── test_components.py
├── main.py                     # Main entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Run with default webcam
python main.py

# Run Streamlit UI
streamlit run ui/app.py
```

### 3. Train Your Own Model

```bash
python training/train.py --config config/config.yaml
```

## 📊 System Requirements

- Python 3.8+
- Webcam
- 8GB RAM minimum (16GB recommended)
- GPU optional but recommended for training

## 🎯 Features

- [x] Real-time webcam capture at 30 FPS
- [x] MediaPipe hand and pose landmark detection
- [x] ASL alphabet recognition (A-Z)
- [x] Word-level recognition (common signs)
- [x] Text-to-speech output
- [x] Clean Streamlit UI
- [ ] Continuous sentence translation
- [ ] Multi-language support

## 📚 Documentation

See individual module documentation in their respective folders.

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Camera settings
- Model parameters
- TTS voice settings
- UI preferences

## 📄 License

MIT License - See LICENSE file
