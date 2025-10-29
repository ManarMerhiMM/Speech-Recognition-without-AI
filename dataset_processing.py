# Purpose: Preprocess all WAVs in ./Data the same way the main script does
# (DC offset removal, normalization, noise gating).
# Saves cleaned copies in ./Processed (sibling of /Data, not inside it).

import os
import wave
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUT_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")  # <-- CHANGED
os.makedirs(OUT_DIR, exist_ok=True)

RATE = 16000  # target sample rate (Hz)

def read_wav_as_float64(path: str) -> np.ndarray:
    """Read mono WAV as float64."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        sig = sig[::n_channels]

    if framerate != RATE and sig.size:
        x_old = np.linspace(0, 1, num=sig.size, endpoint=False)
        new_len = int(round(sig.size * (RATE / float(framerate))))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        sig = np.interp(x_new, x_old, sig)
    return sig

def preprocess_signal(sig: np.ndarray, noise_gate: float = 0.02) -> np.ndarray:
    """Remove DC offset, normalize amplitude, apply simple noise gate."""
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    sig[np.abs(sig) < noise_gate] = 0.0
    return sig

def write_wav(path: str, sig: np.ndarray, rate: int = RATE):
    """Write float64 signal as 16-bit PCM WAV."""
    sig = np.clip(sig, -1.0, 1.0)
    int16_data = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(int16_data.tobytes())

# === Process all .wav files ===
for fn in os.listdir(DATA_DIR):
    if fn.lower().endswith(".wav"):
        src = os.path.join(DATA_DIR, fn)
        dst = os.path.join(OUT_DIR, fn)
        print(f"Processing: {fn}")
        try:
            sig = read_wav_as_float64(src)
            sig = preprocess_signal(sig)
            write_wav(dst, sig)
        except Exception as e:
            print(f"  Failed on {fn}: {e}")

print(f"✅ Done. Processed files saved in: {OUT_DIR}")