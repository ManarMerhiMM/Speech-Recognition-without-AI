# script.py — Terminal matcher (fast): loads cached references, records at 8 kHz, FFT-based NCC, Windows-safe multiprocessing, and TTS reply.
# Note: If a filename ends with digits (e.g., "Hello2"), the phrase is recognized without the numeric suffix.

import os
import re
import sys
import time
import numpy as np
import sounddevice as sd
import pyttsx3
from scipy.signal import fftconvolve
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# -----------------------------
# Config
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "Preprocessed", "ref_cache.npz")

RATE = 8000                 # must match dataset_processing.py cache rate
DURATION = 3.0              # seconds to record
SIMILARITY_THRESHOLD = 0.12 # tune per dataset (e.g., 0.12–0.20)

# -----------------------------
# Utils
# -----------------------------
def trim_silence(x: np.ndarray, thr: float = 0.02, win: int = 1024) -> np.ndarray:
    """Trim leading/trailing silence using RMS threshold."""
    if x.size == 0:
        return x
    x = x.astype(np.float32, copy=False)
    sq = x * x
    kernel = np.ones(win, dtype=np.float32) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + 1e-8)
    mask = rms > thr
    if not np.any(mask):
        return x
    idx = np.where(mask)[0]
    return x[idx[0]: idx[-1] + 1]

def preprocess_signal(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, peak-norm to 1, trim silence; keep float32."""
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    m = np.max(np.abs(sig)) + 1e-12
    sig = (sig / m).astype(np.float32)
    return trim_silence(sig, thr=0.02, win=1024)

def ncc_fft_max(long_sig: np.ndarray, short_sig: np.ndarray) -> float:
    """Max normalized cross-correlation using FFT convolution (fast)."""
    L = len(short_sig)
    if L == 0 or len(long_sig) < L:
        return 0.0
    corr = fftconvolve(long_sig, short_sig[::-1], mode="valid")
    ss = np.sum(short_sig * short_sig, dtype=np.float64)
    if ss <= 0.0:
        return 0.0
    short_norm = np.sqrt(ss)
    lsq = long_sig * long_sig
    csum = np.cumsum(lsq, dtype=np.float64)
    win_energy = csum[L-1:] - np.concatenate(([0.0], csum[:-L]))
    denom = short_norm * np.sqrt(np.maximum(win_energy, 1e-12))
    coeff = corr / denom
    if coeff.size == 0:
        return 0.0
    return float(np.max(coeff))

def score_one(args) -> float:
    """Score pair (rec, ref) → max NCC."""
    rec, ref = args
    if len(rec) >= len(ref):
        long_sig, short_sig = rec, ref
    else:
        long_sig, short_sig = ref, rec
    return ncc_fft_max(long_sig, short_sig)

def load_cache(path: str):
    """Load preprocessed reference cache produced by dataset_processing.py."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cache not found: {path}\nRun: python dataset_processing.py\n"
            "This generates Preprocessed/ref_cache.npz"
        )
    z = np.load(path, allow_pickle=True)
    data = {
        "speakers": z["speakers"],
        "phrases": z["phrases"],
        "filenames": z["filenames"],
        "signals": z["signals"],
        "rate": int(z["rate"]),
    }
    return data

def normalize_phrase(phrase_raw: str) -> str:
    """Strip any trailing digits (e.g., 'Hello2' -> 'Hello')."""
    return re.sub(r"\s*\d+\s*$", "", str(phrase_raw)).strip()

def build_response(speaker: str | None, phrase_raw: str | None) -> str:
    """Map phrase→reply; default echo if unknown."""
    if not speaker or not phrase_raw:
        return "Sorry, I didn't catch that."
    phrase_key = normalize_phrase(phrase_raw).lower()
    if phrase_key == "hello":
        return f"Hey! {speaker}"
    if phrase_key == "how are you":
        return f"I am fine thank you, {speaker}!"
    return f"You said: {normalize_phrase(phrase_raw)}, {speaker}."

# -----------------------------
# Main
# -----------------------------
def main():
    ref = load_cache(CACHE_PATH)
    if ref["rate"] != RATE:
        raise RuntimeError(f"Cache rate {ref['rate']} != recorder rate {RATE}. Re-run dataset_processing.py.")

    print("Listening for an audio phrase...")
    rec = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="float32")
    sd.wait()
    rec = preprocess_signal(rec.flatten())
    print("Finished Listening, started processing...")

    t0 = time.perf_counter()
    scores: list[float] = []
    try:
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
            scores = list(ex.map(score_one, ((rec, s) for s in ref["signals"]), chunksize=8))
    except RuntimeError:
        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(score_one, ((rec, s) for s in ref["signals"])))

    best_idx = int(np.argmax(scores)) if scores else -1
    best_score = scores[best_idx] if best_idx >= 0 else -1.0
    elapsed = time.perf_counter() - t0

    if best_idx >= 0 and best_score >= SIMILARITY_THRESHOLD:
        spk = str(ref["speakers"][best_idx])
        phr_raw = str(ref["phrases"][best_idx])
        phr = normalize_phrase(phr_raw)
        fname = str(ref["filenames"][best_idx])
        response = build_response(spk, phr)
    else:
        spk = phr = fname = None
        response = "Sorry, I didn't catch that."

    print("Response:", response)
    print(f"Max correlation coefficient: {best_score:.3f}")
    if best_idx >= 0:
        print(f"Best match file: {fname}")
        print(f"Recognized phrase (normalized): {phr}")
    print(f"Total correlation time: {elapsed:.2f} s")

    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# -----------------------------
# Windows-safe entry point
# -----------------------------
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")