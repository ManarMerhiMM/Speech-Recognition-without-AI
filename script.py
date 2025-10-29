import os
import wave
import numpy as np
import sounddevice as sd
import pyttsx3

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")
RATE = 16000
DURATION = 3  # seconds
SIMILARITY_THRESHOLD = 0.15


def read_wav_as_float64(path: str, target_rate: int) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if n_channels > 1:
        sig = sig[::n_channels]
    if framerate != target_rate and sig.size > 0:
        duration = n_frames / float(framerate)
        old_time = np.linspace(0, duration, num=sig.size, endpoint=False)
        new_time = np.linspace(0, duration, num=int(round(duration * target_rate)), endpoint=False)
        sig = np.interp(new_time, old_time, sig)
    return sig


def preprocess_signal(sig: np.ndarray, noise_gate: float = 0.02) -> np.ndarray:
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    sig[np.abs(sig) < noise_gate] = 0.0
    return sig


def znorm_xcorr_max(long_signal: np.ndarray, short_signal: np.ndarray) -> float:
    L = len(short_signal)
    if L == 0 or len(long_signal) < L:
        return 0.0
    corr = np.correlate(long_signal, short_signal, mode="valid")
    short_energy = np.sum(short_signal * short_signal)
    if short_energy == 0:
        return 0.0
    short_norm = np.sqrt(short_energy)
    lsq = long_signal * long_signal
    csum = np.cumsum(lsq)
    win_energy = csum[L-1:] - np.concatenate(([0.0], csum[:-L]))
    denom = short_norm * np.sqrt(np.maximum(win_energy, 1e-12))
    coeff = corr / denom
    coeff = coeff[np.isfinite(coeff)]
    if coeff.size == 0:
        return 0.0
    return float(np.max(coeff))


# --- Record audio using sounddevice ---
print("Listening for an audio phrase...")
recorded_samples = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype='float64')
sd.wait()
recorded_samples = recorded_samples.flatten()
recorded_samples = preprocess_signal(recorded_samples)

# --- Load and preprocess reference signals ---
if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"Reference directory not found: {DATA_DIR}")

reference_signals = []
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".wav"):
        try:
            speaker, phrase_with_ext = filename.split(" - ", 1)
            phrase = os.path.splitext(phrase_with_ext)[0]
        except ValueError:
            print(f"Skipping bad filename: {filename}")
            continue
        filepath = os.path.join(DATA_DIR, filename)
        samples = read_wav_as_float64(filepath, target_rate=RATE)
        samples = preprocess_signal(samples)
        reference_signals.append((speaker, phrase, samples))

if not reference_signals:
    print("No reference WAV files found. Please preprocess your dataset.")
    exit(1)

# --- Match recorded audio to reference samples ---
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

# --- Respond based on result ---
if best_match is not None and best_score >= SIMILARITY_THRESHOLD:
    identified_speaker, identified_phrase = best_match
    phrase_key = identified_phrase.strip().lower()
    if phrase_key == "hello":
        response_text = f"Hey! {identified_speaker}"
    elif phrase_key == "how are you":
        response_text = f"I am fine thank you, {identified_speaker}!"
    else:
        response_text = f"You said: {identified_phrase}, {identified_speaker}."
else:
    response_text = "Sorry, I didn't catch that."

print("Response:", response_text)

# --- Speak the response ---
engine = pyttsx3.init()
engine.say(response_text)
engine.runAndWait()
