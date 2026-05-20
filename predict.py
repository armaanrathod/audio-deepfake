import torch
import librosa
import numpy as np
from model import DeepfakeCNN

# Load model once when the file is imported
model = DeepfakeCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu", weights_only=True))
model.eval()

SR = 16000
DURATION = 4.0
SLAB_LENGTH = int(SR * DURATION)  # 64000 samples per slab


def _spectrogram_from_array(y: np.ndarray) -> torch.Tensor:
    """Convert a numpy audio array to a model-ready tensor."""
    # Pad if shorter than 4 seconds
    if len(y) < SLAB_LENGTH:
        y = np.pad(y, (0, SLAB_LENGTH - len(y)))

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Slice then pad to guarantee exactly (128, 126) regardless of librosa version
    arr = mel_db[:128, :126]
    if arr.shape[1] < 126:
        arr = np.pad(arr, ((0, 0), (0, 126 - arr.shape[1])), constant_values=arr.min())
    spec = torch.tensor(arr, dtype=torch.float32)
    return spec.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 126)


def predict(audio_path: str) -> dict:
    """
    Main prediction function.
    Accepts any length audio — splits into 4-second slabs,
    predicts on each, and averages the results.
    """
    # Load full audio at 16kHz mono — no duration cap
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    total_duration = len(y) / SR

    # Split into 4-second slabs
    slabs = []
    for i in range(0, len(y), SLAB_LENGTH):
        slab = y[i:i + SLAB_LENGTH]
        slabs.append(slab)  # _spectrogram_from_array handles padding

    # Predict on each slab
    slab_probs = []
    for slab in slabs:
        spec = _spectrogram_from_array(slab)
        with torch.no_grad():
            prob = model(spec).item()
        slab_probs.append(prob)

    if not slab_probs:
        raise ValueError("Audio file is empty or could not be decoded.")

    # Average probability across all slabs
    avg_prob = sum(slab_probs) / len(slab_probs)

    label = "fake" if avg_prob > 0.5 else "real"
    confidence = round(avg_prob if avg_prob > 0.5 else 1 - avg_prob, 3)

    return {
        "prediction": label,
        "confidence": confidence,
        "slabs_analysed": len(slab_probs),
        "audio_duration_seconds": round(total_duration, 1)
    }
