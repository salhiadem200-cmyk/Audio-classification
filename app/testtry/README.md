# Real-Time Sound Detection App

This is a small Python app that detects sound in real time using your microphone.

## Requirements
- Python 3.7+
- The following Python libraries:
  - sounddevice
  - librosa
  - numpy
  - scikit-learn
  - lightgbm
  - joblib

## Setup
1. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
1. Run the main script:
   ```bash
   python main.py
   ```

## Notes
- Make sure your microphone is connected and accessible.
- This app is a template for real-time sound detection. You can extend it to classify or process sounds as needed. 