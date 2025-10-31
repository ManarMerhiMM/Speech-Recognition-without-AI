# dataset_processing.py — Preprocess WAVs to 8 kHz mono, trim silence, normalize, and build a fast .npz cache

import os, wave
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
OUT_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")
CACHE_PATH = os.path.join(OUT_DIR, "ref_cache.npz")
os.makedirs(OUT_DIR, exist_ok=True)

RATE = 8000  # 8 kHz is plenty for speech matching

def read_wav_float32(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        fr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        sig = sig[::ch]
    if fr != RATE and sig.size > 0:
        dur = n / float(fr)
        old_t = np.linspace(0, dur, num=sig.size, endpoint=False, dtype=np.float32)
        new_t = np.linspace(0, dur, num=int(round(dur * RATE)), endpoint=False, dtype=np.float32)
        sig = np.interp(new_t, old_t, sig).astype(np.float32)
    return RATE, sig

def trim_silence(x: np.ndarray, thr: float = 0.02, win: int = 1024) -> np.ndarray:
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
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    m = np.max(np.abs(sig)) + 1e-12
    sig = (sig / m).astype(np.float32)
    sig = trim_silence(sig, thr=0.02, win=1024)
    return sig

def write_wav(path: str, sig: np.ndarray, rate: int = RATE):
    sig16 = np.clip(sig, -1.0, 1.0)
    sig16 = (sig16 * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(sig16.tobytes())

meta = []       # list of (speaker, phrase, filename)
signals = []    # list of float32 arrays

for filename in os.listdir(DATA_DIR):
    if not filename.lower().endswith(".wav"):
        continue
    src = os.path.join(DATA_DIR, filename)
    try:
        spk, rest = filename.split(" - ", 1)
        phrase = os.path.splitext(rest)[0]
        # strip trailing numbers like "Hello 2"
        phrase = phrase.rsplit(" ", 1)[0] if phrase.split() and phrase.split()[-1].isdigit() else phrase
    except ValueError:
        print(f"Skipping bad filename: {filename}")
        continue

    print(f"Processing: {filename}")
    _, sig = read_wav_float32(src)
    sig = preprocess_signal(sig)

    dst = os.path.join(OUT_DIR, filename)
    write_wav(dst, sig)
    meta.append((spk, phrase, filename))
    signals.append(sig)

# Save fast cache (object arrays to allow variable lengths)
np.savez_compressed(
    CACHE_PATH,
    speakers=np.array([m[0] for m in meta], dtype=object),
    phrases=np.array([m[1] for m in meta], dtype=object),
    filenames=np.array([m[2] for m in meta], dtype=object),
    signals=np.array(signals, dtype=object),
    rate=np.array(RATE, dtype=np.int32),
)

print(f"✅ Done. WAVs saved in: {OUT_DIR}")
print(f"⚡ Cache saved: {CACHE_PATH}")