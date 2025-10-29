# Purpose: Fast voice matching demo with basic denoising and normalization.
# Records 3s at 16 kHz, loads reference WAVs from ./Preprocessed,
# applies preprocessing (DC offset removal, normalization, noise gating),
# computes normalized cross-correlation (vectorized), and speaks a response.

import os
import wave
import numpy as np
import pyaudio
import pyttsx3

# === Paths / config ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")  # <-- CHANGED

# === Audio recording config ===
RATE = 16000
CHANNELS = 1
CHUNK = 1024
RECORD_SECONDS = 3

# === Utility functions ===
def read_wav_as_float64(path: str, target_rate: int) -> np.ndarray:
    """Read a WAV file and return mono float64 samples at target_rate."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        sig = sig[::n_channels]

    if framerate != target_rate and sig.size:
        x_old = np.linspace(0, 1, num=sig.size, endpoint=False)
        new_len = int(round(sig.size * (target_rate / float(framerate))))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        sig = np.interp(x_new, x_old, sig)
    return sig

def preprocess_signal(sig: np.ndarray, noise_gate: float = 0.02) -> np.ndarray:
    """Remove DC offset, normalize amplitude, apply a simple noise gate."""
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    sig[np.abs(sig) < noise_gate] = 0.0
    return sig

def znorm_xcorr_max(long_signal: np.ndarray, short_signal: np.ndarray) -> float:
    """Return maximum normalized cross-correlation between long and short 1D signals."""
    L = len(short_signal)
    if L == 0 or len(long_signal) < L:
        return 0.0

    short_energy = np.sum(short_signal * short_signal)
    if short_energy <= 0:
        return 0.0
    short_norm = np.sqrt(short_energy)

    corr = np.correlate(long_signal, short_signal, mode="valid")
    lsq = long_signal * long_signal
    csum = np.cumsum(lsq)
    win_energy = csum[L - 1:] - np.concatenate(([0.0], csum[:-L]))
    denom = short_norm * np.sqrt(np.maximum(win_energy, 1e-12))

    coeff = corr / denom
    coeff = coeff[np.isfinite(coeff)]
    return float(np.max(coeff)) if coeff.size else 0.0

# === Record from mic ===
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("Listening for an audio phrase...")
frames = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    frames.append(stream.read(CHUNK, exception_on_overflow=False))
print("Recording complete.")
stream.stop_stream()
stream.close()
p.terminate()

# === Convert and preprocess recorded audio ===
recorded_data = b"".join(frames)
recorded_samples = np.frombuffer(recorded_data, dtype=np.int16).astype(np.float64)
recorded_samples = preprocess_signal(recorded_samples)

# === Load and preprocess reference WAVs ===
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Directory 'Preprocessed' not found at: {DATA_DIR}")

reference_signals = []
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".wav"):
        try:
            speaker, phrase_with_ext = filename.split(" - ", 1)
            phrase = os.path.splitext(phrase_with_ext)[0]
        except ValueError:
            print(f"Skipping invalid filename: {filename}")
            continue

        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        print("Loading:", filepath)
        try:
            samples = read_wav_as_float64(filepath, target_rate=RATE)
            samples = preprocess_signal(samples)
            reference_signals.append((speaker, phrase, samples))
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

if not reference_signals:
    print("No reference WAVs found in Preprocessed folder.")
    response_text = "Sorry, I didn't catch that."
    print("Detected: None (score: -1.00)")
    print("Response:", response_text)
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()
    raise SystemExit(0)

# === Matching ===
SIMILARITY_THRESHOLD = 0
best_match = None
best_score = -1.0

for speaker, phrase, ref_signal in reference_signals:
    if len(recorded_samples) >= len(ref_signal):
        long_sig, short_sig = recorded_samples, ref_signal
    else:
        long_sig, short_sig = ref_signal, recorded_samples

    score = znorm_xcorr_max(long_sig, short_sig)
    if score > best_score:
        best_score = score
        best_match = (speaker, phrase)

# === Response ===
if best_match and best_score >= SIMILARITY_THRESHOLD:
    speaker, phrase = best_match
    key = phrase.strip().lower()
    if key == "hello":
        response_text = f"Hey! {speaker}"
    elif key == "how are you":
        response_text = f"I am fine thank you {speaker}!"
    else:
        response_text = f"You said: {phrase}, {speaker}."
else:
    response_text = "Sorry, I didn't catch that."

print(f"Detected: {best_match} (score: {best_score:.2f})")
print("Response:", response_text)

# === Speak ===
engine = pyttsx3.init()
engine.say(response_text)
engine.runAndWait()