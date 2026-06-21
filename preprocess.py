import librosa
import numpy as np
import os
import pandas as pd

# Paths
AUDIO_DIR = "C:\\Users\\talk2\\OneDrive\\Desktop\\Deepfake Audio\\Data\\LA\\LA\\ASVspoof2019_LA_dev\\flac"

PROTOCOL_FILE = "C:\\Users\\talk2\\OneDrive\\Desktop\\Deepfake Audio\\Data\\LA\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.dev.trl.txt"

OUTPUT_DIR = "./spectrograms"

# Function 1 - Load audio
def load_audio(file_path, sr=16000, duration=4.0):
    audio, _ = librosa.load(file_path, sr=sr)
    target_length = int(sr * duration)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    return audio

# Convert to spectrogram
def audio_to_melspectrogram(audio, sr=16000):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        fmax=8000,
        hop_length=512,
        n_fft=1024
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function 3 - Save spectrogram
def save_spectrogram(mel_spec_db, file_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_id + ".npy")
    np.save(save_path, mel_spec_db)
    return save_path

# Load labels
df = pd.read_csv(
    PROTOCOL_FILE,
    sep=" ",
    header=None,
    names=["speaker_id", "file_id", "env", "attack_type", "label"]
)
df["label_num"] = df["label"].map({"bonafide": 0, "spoof": 1})

results = []
print(f"Starting processing of {len(df)} files...")

for idx, row in df.iterrows():
    file_path = os.path.join(AUDIO_DIR, row["file_id"] + ".flac")
    if not os.path.exists(file_path):
        continue
    try:
        audio     = load_audio(file_path)
        mel_spec  = audio_to_melspectrogram(audio)
        save_path = save_spectrogram(mel_spec, row["file_id"], OUTPUT_DIR)
        results.append({
            "file_id":   row["file_id"],
            "spec_path": save_path,
            "label":     row["label_num"]
        })
    except Exception as e:
        print(f"Error on {row['file_id']}: {e}")
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} files done...")

label_df = pd.DataFrame(results)
label_df.to_csv("./labels.csv", index=False)
print(f"\nALL DONE!")
print(f"Spectrograms saved: {len(label_df)}")
print(f"Label file saved:   ./labels.csv")
print(label_df["label"].value_counts())
