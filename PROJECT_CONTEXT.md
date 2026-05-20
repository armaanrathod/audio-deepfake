# Audio Deepfake Detector — Project Context
**Last updated:** 18 May 2026
**Team:** Armaan (ML), Dhiya (Data), Navya (Backend), Vania (Frontend)
**Submission deadline:** 21 May 2026

---

## What this project does

Classifies any voice/audio clip as either **REAL** (human recording) or **FAKE** (AI-synthesised voice).

End-to-end flow:
```
User uploads audio (wav/mp3/flac/m4a/ogg)
  → Streamlit frontend (frontend.py, port 8501)
  → POST /predict to Flask backend (app.py, port 5000)
  → predict.py loads audio, splits into 4-second slabs, converts each to mel spectrogram
  → CNN model (model.py) scores each slab → probabilities averaged
  → JSON result returned → Streamlit shows REAL/FAKE + confidence %
```

---

## Files and what each one does

| File | Owner | Purpose |
|------|-------|---------|
| `model.py` | Armaan | CNN architecture — `DeepfakeCNN` class |
| `train.py` | Armaan | Training script — run on Google Colab, not locally |
| `predict.py` | Armaan | Preprocessing + inference — called by `app.py` |
| `app.py` | Navya | Flask REST API (backend) |
| `frontend.py` | Vania | Streamlit UI (frontend) |
| `model.pth` | Armaan | Saved model weights — download from Colab after training |
| `download.py` | Dhiya | Downloads training data folder from Google Drive using `gdown` |
| `labels_balanced.csv` | Dhiya | Training index (stays on Colab/Drive) |
| `spectrograms/` | Dhiya | `.npy` training files (stays on Colab/Drive) |

**To run locally you only need:** `model.py`, `predict.py`, `app.py`, `frontend.py`, `model.pth`

---

## Dataset

**Source:** ASVspoof 2019 LA (Logical Access) — dev partition
**Raw split:** 1,710 real + 14,913 fake (heavily imbalanced)
**After balancing:** 14,913 real + 14,913 fake = **29,826 total**

Balancing was done by augmenting real samples with three cyclically applied transformations:
- Gaussian noise (std=0.5 added to spectrogram values)
- Time shift (±5 columns)
- Frequency shift (±2 rows)

**Spectrogram format:**
- Sample rate: 16,000 Hz
- Clip duration: 4 seconds → 64,000 samples
- Shape: (128, 126) — 128 mel bands × 126 time frames
- Scale: Decibels via `librosa.power_to_db(ref=np.max)` → roughly −80 to 0
- Stored as `.npy` numpy arrays

**CSV columns:** `spec_path` (path to .npy), `label` (0 = real, 1 = fake)

---

## Model architecture (`model.py`)

Class: `DeepfakeCNN(nn.Module)`

```
Input:  (batch, 1, 128, 126)

Conv2d(1→32, 3×3, pad=1) + ReLU + MaxPool2d(2)   → (batch, 32, 64, 63)
Conv2d(32→64, 3×3, pad=1) + ReLU + MaxPool2d(2)  → (batch, 64, 32, 31)
Conv2d(64→128, 3×3, pad=1) + ReLU + MaxPool2d(2) → (batch, 128, 16, 15)

Flatten → 30,720  (128 × 16 × 15)
Linear(30720 → 256) + ReLU + Dropout(0.5)
Linear(256 → 1) + Sigmoid

Output: single float in [0, 1]
  > 0.5 → FAKE
  ≤ 0.5 → REAL
```

**Rationale:** Once audio is converted to a spectrogram it is a 2D image — CNNs are the natural fit. Binary classification (real vs fake) is structurally identical to any two-class image problem.

**Training config (run on Colab T4 GPU):**
- Epochs: 10
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross Entropy (`nn.BCELoss`)
- Reported training accuracy: 99.8%, training loss: 0.009
- Device: auto-detected (`cuda` if available, else `cpu`) — model and batches both moved to device

Note: 99.8% is on training data — not a measure of real-world accuracy.

---

## Preprocessing pipeline (`predict.py`)

The model is loaded once at module import time (`map_location="cpu"`) and set to `eval()` mode — it stays resident for the lifetime of the Flask process.

For each prediction call:

```
1. librosa.load(path, sr=16000, mono=True)
   — resample to 16kHz, force single channel, no duration cap

2. Split into 4-second slabs (64,000 samples each)
   — iterates with step=SLAB_LENGTH over the full waveform

3. For each slab:
   a. Pad with silence if shorter than 64,000 samples
   b. librosa.feature.melspectrogram(n_mels=128, hop_length=512)
   c. librosa.power_to_db(mel, ref=np.max)
   d. Slice to [:128, :126] then pad time axis to 126 if short — guarantees exact (128, 126)
   e. torch.tensor().unsqueeze(0).unsqueeze(0) → (1, 1, 128, 126)
   f. model(spec).item() → slab probability (torch.no_grad())

4. Guard: raises ValueError if slab_probs is empty (corrupt/zero-length file)

5. avg_prob = mean of all slab probabilities

6. label = "fake" if avg_prob > 0.5 else "real"
   confidence = avg_prob if fake, else (1 − avg_prob) if real

Return dict: {prediction, confidence, slabs_analysed, audio_duration_seconds}
```

---

## Backend (`app.py`)

Framework: Flask, port 5000

**POST `/predict`**
- Accepts `multipart/form-data` with field `file`
- Allowed extensions: `wav`, `mp3`, `flac`, `m4a`, `ogg`
- Saves upload to a `tempfile.NamedTemporaryFile`, calls `predict()`, deletes the temp file in a `finally` block (existence-checked before unlink)
- Returns JSON: `{prediction, confidence, slabs_analysed, audio_duration_seconds}`
- Error responses: 400 for missing/empty file or bad format, 500 for exceptions

**GET `/health`**
- Returns `{"status": "ok", "model": "loaded"}`

Runs with `debug=False`.

---

## Frontend (`frontend.py`)

Framework: Streamlit, auto port 8501

- File uploader (wav, mp3, flac, m4a, ogg)
- Audio playback widget (plays uploaded file before analysis)
- "Analyze" primary button
- Writes upload to a temp file, closes it fully before opening for the POST request (Windows file-locking workaround)
- On success: shows REAL (green `st.success`) or FAKE (red `st.error`)
- Three metric columns: Confidence %, Duration (s), Slabs analysed
- Progress bar (`st.progress`) at confidence value
- Caption explaining slab count
- `requests.post` uses a 120-second timeout to avoid hanging on large files
- Checks `resp.ok` before calling `resp.json()` — shows raw status + text on non-2xx instead of crashing with `JSONDecodeError`
- On `ConnectionError`: tells user to start `app.py` on port 5000

---

## Training data download (`download.py`)

Uses `gdown` to download the full Google Drive folder containing spectrograms and the balanced CSV. Only needed when setting up the Colab training environment — not required for local inference.

---

## How to run locally

Requires: `model.py`, `predict.py`, `app.py`, `frontend.py`, `model.pth` in the same directory.

```bash
pip install torch torchaudio librosa numpy flask streamlit requests
```

Terminal 1 — backend:
```bash
python app.py
# Running on http://127.0.0.1:5000
```

Terminal 2 — frontend:
```bash
streamlit run frontend.py
# Opens http://localhost:8501
```

---

## Bug fixes (18 May 2026)

Eight bugs found and fixed during stress testing:

| # | File | Bug | Impact |
|---|------|-----|--------|
| 1 | `predict.py` | `torch.load` missing `weights_only=True` | Hard error on PyTorch ≥ 2.6 |
| 2 | `predict.py` | Empty/corrupt audio → `slab_probs = []` → `ZeroDivisionError` | Crash on any zero-length file |
| 3 | `predict.py` | Spectrogram only sliced, not padded to (128, 126) — model shape mismatch if librosa returns <126 frames | Crash on edge-case frame count |
| 4 | `app.py` | `os.unlink(tmp_path)` with no existence guard in `finally` | `FileNotFoundError` in edge cases |
| 5 | `train.py` | No GPU device handling — model and batches never moved to CUDA | Colab T4 training silently ran on CPU (10× slower) |
| 6 | `train.py` | `.squeeze()` instead of `.squeeze(1)` on model output | `BCELoss` shape mismatch when last batch size = 1 |
| 7 | `frontend.py` | No `timeout` on `requests.post` | UI hangs forever on slow/stalled backend |
| 8 | `frontend.py` | `resp.json()` called before checking `resp.ok` | `JSONDecodeError` on Flask HTML error pages |

---

## Known limitations

1. **Training accuracy ≠ real-world accuracy.** 99.8% was measured on training data the model already saw.
2. **Background music degrades performance.** Music adds broadband frequency content that overlaps with voice patterns.
3. **Noisy recordings.** Room noise, echo, wind, or phone compression can cause misclassification. The model expects clean 16kHz mono audio.
4. **Out-of-distribution AI voices.** Trained on ASVspoof 2019 synthesis methods. Newer tools (ElevenLabs, etc.) use different techniques the model has never seen.

---

## Planned improvements

- [ ] Retrain on YouTube and real-world noisy data
- [ ] Spectrogram visualisation in the frontend (Phase 2)
- [ ] Per-slab confidence bar (show which segment triggered the result)
- [ ] YouTube URL input for longer clips

---

## Retraining

Training only runs on Colab — do not run `train.py` locally (no GPU, and training data lives on Drive).

```
1. Open Colab, mount Drive
2. Run: python train.py
3. Download model.pth from Colab file browser before session ends
4. Replace model.pth in local project folder
```
