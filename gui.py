# gui.py — Tkinter GUI that mirrors script.py logic & outputs: loads cached refs (Preprocessed/ref_cache.npz),
# records @ 8 kHz, FFT-based NCC matching, thresholded decision, and TTS reply. Displays the same fields
# (Response, Max corr, Best match file, Recognized phrase (normalized), Total correlation time).
# Update: buttons use pointer cursor. TTS is reliable: a NEW pyttsx3 engine is created/used per recording
# inside the same worker thread (Windows/SAPI-safe), after ensuring the audio device is free).
# Theme: Midnight Gold (dark, black/white with gold accents).

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os, time, re

import numpy as np
import sounddevice as sd
from scipy.signal import fftconvolve
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Config (aligned with script.py)
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "Preprocessed", "ref_cache.npz")
PREPROCESSED_DIR = os.path.join(SCRIPT_DIR, "Preprocessed")

RATE = 8000                 # must match dataset_processing.py cache rate
RECORD_SECONDS = 3.0        # seconds to record
SIMILARITY_THRESHOLD = 0.12 # tune per dataset (e.g., 0.12–0.20)
RECORD_FILE = os.path.join(SCRIPT_DIR, "record_latest.wav")

# -----------------------------
# State
# -----------------------------
recorded_data = None
recorded_path = None
best_match_path = None

# Cache (loaded once at startup)
ref = None  # dict with keys: speakers, phrases, filenames, signals, rate

# -----------------------------
# Matching utils (identical logic to script.py)
# -----------------------------
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
    return trim_silence(sig, thr=0.02, win=1024)

def ncc_fft_max(long_sig: np.ndarray, short_sig: np.ndarray) -> float:
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
    rec_sig, ref_sig = args
    if len(rec_sig) >= len(ref_sig):
        long_sig, short_sig = rec_sig, ref_sig
    else:
        long_sig, short_sig = ref_sig, rec_sig
    return ncc_fft_max(long_sig, short_sig)

def normalize_phrase(phrase_raw: str) -> str:
    return re.sub(r"\s*\d+\s*$", "", str(phrase_raw)).strip()

def build_response(speaker: str | None, phrase_raw: str | None) -> str:
    if not speaker or not phrase_raw:
        return "Sorry, I didn't catch that."
    phrase_key = normalize_phrase(phrase_raw).lower()
    if phrase_key == "hello":
        return f"Hey! {speaker}"
    if phrase_key == "how are you":
        return f"I am fine thank you, {speaker}!"
    return f"You said: {normalize_phrase(phrase_raw)}, {speaker}."

def load_cache(path: str):
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

# -----------------------------
# Button actions
# -----------------------------
def start_recording():
    record_btn.config(state="disabled")
    status_var.set("Listening for an audio phrase...")
    threading.Thread(target=_record_and_match_flow, daemon=True).start()

def _record_and_match_flow():
    global recorded_data, recorded_path, best_match_path
    try:
        # Record
        rec = sd.rec(int(RECORD_SECONDS * RATE), samplerate=RATE, channels=1, dtype="float32")
        sd.wait()
        recorded_data = rec.flatten()
        recorded_data = preprocess_signal(recorded_data)

        # Save last recording (processed) for relistening
        try:
            import wave
            sig16 = np.clip(recorded_data, -1.0, 1.0)
            sig16 = (sig16 * 32767.0).astype(np.int16)
            with wave.open(RECORD_FILE, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(sig16.tobytes())
            recorded_path = RECORD_FILE
        except Exception:
            recorded_path = None

        status_var.set("Finished Listening, started processing...")

        # Score in parallel (threads avoid Windows spawn issues in GUI)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
            scores = list(ex.map(score_one, ((recorded_data, s) for s in ref["signals"]), chunksize=32))
        best_idx = int(np.argmax(scores)) if scores else -1
        best_score = scores[best_idx] if best_idx >= 0 else -1.0
        elapsed = time.perf_counter() - t0

        # Decide match
        if best_idx >= 0 and best_score >= SIMILARITY_THRESHOLD:
            spk = str(ref["speakers"][best_idx])
            phr_raw = str(ref["phrases"][best_idx])
            phr = normalize_phrase(phr_raw)
            fname = str(ref["filenames"][best_idx])
            response = build_response(spk, phr)
            # prefer preprocessed file for playback
            ppath = os.path.join(PREPROCESSED_DIR, fname)
            best_match_path = ppath if os.path.isfile(ppath) else None
        else:
            spk = phr = fname = None
            response = "Sorry, I didn't catch that."
            best_match_path = None

        # UI: mirror script.py outputs
        status_var.set(f"Finished correlation ({elapsed:.2f} s)")

        if best_idx >= 0:
            best_match_var.set(f"Best match file: {fname} | Max correlation coefficient: {best_score:.3f}")
        else:
            best_match_var.set("Best match file: — | Max correlation coefficient: 0.000")

        if best_idx >= 0 and spk and phr:
            reply_lines = [
                f"Response: {response}",
                f"Recognized phrase (normalized): {phr}"
            ]
        else:
            reply_lines = [f"Response: {response}"]
        reply_var.set("\n".join(reply_lines))

        # --- RELIABLE TTS: speak here, in THIS worker thread, with a fresh engine ---
        try:
            sd.stop()  # ensure device isn't in use
        except Exception:
            pass
        try:
            import pyttsx3
            tts = pyttsx3.init()
            # Optional: tts.setProperty("rate", 180); tts.setProperty("volume", 1.0)
            tts.say(response)
            tts.runAndWait()
            tts.stop()
            del tts
        except Exception as e:
            print("TTS error:", e)
        # ---------------------------------------------------------------------------

    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        record_btn.config(state="normal")

def play_recorded():
    if recorded_data is None:
        messagebox.showinfo("Info", "No recording yet.")
        return
    try:
        sd.stop()
        sd.play(recorded_data, RATE)
    except Exception as e:
        messagebox.showerror("Audio Error", str(e))

def play_best_match():
    if not best_match_path:
        messagebox.showinfo("Info", "No best match yet.")
        return
    try:
        import wave
        with wave.open(best_match_path, "rb") as wf:
            fr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            ch = wf.getnchannels()
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data[::ch]
        sd.stop()
        sd.play(data, fr)
    except Exception as e:
        messagebox.showerror("Audio Error", str(e))

# -----------------------------
# THEME — Midnight Gold (dark, black/white with gold accents)
# -----------------------------
BG = "#121212"        # app background (near-black)
FG = "#FFFFFF"        # primary text
SUB = "#C7C7C7"       # secondary text
HEADER_BG = "#0F0F0F" # header background
HEADER_FG = "#FFD54A" # header text (gold)
BTN_BG = "#1E1E1E"    # button face
BTN_ACTIVE = "#FFD54A" # accent (gold)
BORDER = "#2A2A2A"    # card borders

FONT_BASE = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI Semibold", 14)

# -----------------------------
# UI
# -----------------------------
root = tk.Tk()
root.title("Speech-Recognition-Without-AI")
root.geometry("640x620")
root.resizable(False, False)
root.configure(bg=BG)

style = ttk.Style()
style.theme_use("clam")

# Labels
style.configure("TLabel", background=BG, foreground=FG, font=FONT_BASE)
style.configure("Secondary.TLabel", background=BG, foreground=SUB, font=FONT_BASE)

# Buttons
style.configure("TButton", background=BTN_BG, foreground=FG, font=FONT_BASE, padding=8, borderwidth=1)
style.map("TButton",
          background=[("active", "#2B2B2B"), ("pressed", "#373737")],
          foreground=[("disabled", "#7A7A7A")])

# Labeled frames as “cards”
style.configure("Card.TLabelframe", background=BG, foreground=FG,
                bordercolor=BORDER, relief="solid", borderwidth=1)
style.configure("Card.TLabelframe.Label", background=BG, foreground=HEADER_FG, font=FONT_BOLD)

# Header
header = tk.Frame(root, bg=HEADER_BG, height=52, highlightthickness=0)
header.pack(fill="x")
tk.Label(header, text="Speech Recognition Without AI", bg=HEADER_BG, fg=HEADER_FG, font=FONT_TITLE).pack(pady=10)

container = tk.Frame(root, bg=BG)
container.pack(fill="both", expand=True, padx=16, pady=16)

record_btn = ttk.Button(container, text="🎙  Record", command=start_recording)
record_btn.pack(pady=10)
record_btn.configure(cursor="hand2")  # pointer cursor

status_frame = ttk.LabelFrame(container, text="Output of the User", style="Card.TLabelframe")
status_frame.pack(fill="x", pady=12)
status_frame.configure(borderwidth=1)
status_var = tk.StringVar(value="Waiting for input...")
ttk.Label(status_frame, textvariable=status_var, wraplength=580).pack(padx=12, pady=10, anchor="w")

answer_frame = ttk.LabelFrame(container, text="Answer", style="Card.TLabelframe")
answer_frame.pack(fill="x", pady=12)
answer_frame.configure(borderwidth=1)
best_match_var = tk.StringVar(value="Best match file: — | Max correlation coefficient: 0.000")
reply_var = tk.StringVar(value="Response: —")
ttk.Label(answer_frame, textvariable=best_match_var, wraplength=580).pack(padx=12, pady=(10, 6), anchor="w")
ttk.Label(answer_frame, textvariable=reply_var, wraplength=580, style="Secondary.TLabel", justify="left").pack(padx=12, pady=(0, 10), anchor="w")

audio_block = tk.Frame(container, bg=BG)
audio_block.pack(fill="x", pady=10)

bm_row = tk.Frame(audio_block, bg=BG); bm_row.pack(fill="x", pady=6)
ttk.Label(bm_row, text="Best Match").pack(side="left", padx=(0, 10))
bm_play_btn = ttk.Button(bm_row, text="Play", command=play_best_match)
bm_play_btn.pack(side="left")
bm_play_btn.configure(cursor="hand2")

rec_row = tk.Frame(audio_block, bg=BG); rec_row.pack(fill="x", pady=6)
ttk.Label(rec_row, text="Recorded Signal").pack(side="left", padx=(0, 10))
rec_play_btn = ttk.Button(rec_row, text="Play", command=play_recorded)
rec_play_btn.pack(side="left")
rec_play_btn.configure(cursor="hand2")

# -----------------------------
# Boot: load cache
# -----------------------------
try:
    ref = load_cache(CACHE_PATH)
    if ref["rate"] != RATE:
        raise RuntimeError(f"Cache rate {ref['rate']} != recorder rate {RATE}. Re-run dataset_processing.py.")
except Exception as e:
    messagebox.showerror("Startup Error", str(e))

if __name__ == "__main__":
    root.mainloop()