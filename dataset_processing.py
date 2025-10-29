# dataset_processing.py
# Purpose: Preprocess all WAV files in ./Data and save the cleaned versions in ./Preprocessed.
# This ensures reference audio samples are normalized and denoised similarly to how we'll process live input.

import os
import wave
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUT_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")
os.makedirs(OUT_DIR, exist_ok=True)  # Create Preprocessed directory if it doesn't exist

RATE = 16000  # target sample rate for processing (16 kHz)

def read_wav_as_float64(path: str) -> np.ndarray:
    """Read a WAV file and return mono samples as float64 at RATE Hz."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()  # bytes per sample
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    # Convert bytes to numpy array of int16:
    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    # If stereo, take one channel (downmix to mono)
    if n_channels > 1:
        sig = sig[::n_channels]  # take left channel (for example)
    # Resample to target RATE if needed
    if framerate != RATE and sig.size > 0:
        duration = n_frames / float(framerate)
        # create time axes for original and new signal
        old_time = np.linspace(0, duration, num=sig.size, endpoint=False)
        new_time = np.linspace(0, duration, num=int(duration * RATE), endpoint=False)
        sig = np.interp(new_time, old_time, sig)
    return sig

def preprocess_signal(sig: np.ndarray, noise_gate: float = 0.02) -> np.ndarray:
    """Remove DC offset, normalize amplitude, and apply a simple noise gate."""
    if sig.size == 0:
        return sig  # empty signal, nothing to do
    # Remove DC offset
    sig = sig - np.mean(sig)
    # Normalize to max amplitude 1 (avoid division by zero with small epsilon)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    # Apply noise gate: zero-out low amplitude noise/silence
    sig[np.abs(sig) < noise_gate] = 0.0
    return sig

def write_wav(path: str, sig: np.ndarray, rate: int = RATE):
    """Write a float64 signal array to a WAV file as 16-bit PCM."""
    # Ensure values are in [-1,1] after normalization
    sig_clipped = np.clip(sig, -1.0, 1.0)
    int16_data = (sig_clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)             # mono output
        wf.setsampwidth(2)            # 2 bytes (16 bits) per sample
        wf.setframerate(rate)
        wf.writeframes(int16_data.tobytes())

# Process each WAV file in Data directory
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".wav"):
        src_path = os.path.join(DATA_DIR, filename)
        dst_path = os.path.join(OUT_DIR, filename)
        print(f"Processing: {filename}")
        try:
            signal = read_wav_as_float64(src_path)
            signal = preprocess_signal(signal)
            write_wav(dst_path, signal)
        except Exception as e:
            print(f"  Failed to process {filename}: {e}")

print(f"✅ Done. Processed files are saved in: {OUT_DIR}")
