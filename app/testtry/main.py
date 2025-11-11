import sounddevice as sd
import numpy as np
import librosa
import joblib
import datetime
import os
from termcolor import cprint, colored
import time

# Load your trained model and label encoder
clf = joblib.load("lgbm_sound_classifier.pkl")
le = joblib.load("label_encoder.pkl")

# Sampling settings
sr = 12000          # Sample rate
duration = 1.0      # Duration to record (in seconds)
threshold_class = "gunshot"  # Trigger event if this is detected

# Create a folder to log events
os.makedirs("logs", exist_ok=True)

# ---- Feature Extraction (Same as training) ----
def extract_features_live(y, sr):
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=40)
    log_S = librosa.power_to_db(S)
    spec_mean = np.mean(log_S, axis=1)
    spec_std = np.std(log_S, axis=1)

    return np.concatenate([mfcc_mean, mfcc_std, [zcr_mean, zcr_std], spec_mean, spec_std])

# ---- Main Callback: Called every 1 second ----
# Define the classes you care about
important_classes = {"footsteps", "brokenbranches", "gunshot"}

def callback(indata, frames, time, status):
    if status:
        print("⚠️ Status:", status)

    y = indata[:, 0]  # mono audio

    try:
        features = extract_features_live(y, sr).reshape(1, -1)
        pred = clf.predict(features)[0]
        label = le.inverse_transform([pred])[0].lower()
        prob = clf.predict_proba(features)[0][pred]
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Only process important classes
        if label in important_classes:
            msg = f"[{timestamp}] Detected: {label.upper()} ({prob*100:.1f}%)"
            cprint(msg, 'red' if label == "gunshot" else 'cyan', attrs=['bold'])

            if label == "gunshot":
                with open("logs/detections.txt", "a") as f:
                    f.write(msg + "\n")

    except Exception as e:
        print("❌ Error in prediction:", e)


# ---- Start Listening ----
print(colored("🎧 Starting real-time sound detection. Speak or play audio into your mic...", "green"))
print(colored("🔴 Detected sounds will appear below. Press Ctrl+C to stop.", "yellow"))

try:
    with sd.InputStream(
        callback=callback,
        channels=1,
        samplerate=sr,
        blocksize=int(sr * duration),
        device=1
    ):
        print(colored("🎧 Mic stream active... listening in real-time", "green"))
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print(colored("\n🛑 Stopped by user.", "red"))
except Exception as e:
    print(f"❌ Error: {e}")
