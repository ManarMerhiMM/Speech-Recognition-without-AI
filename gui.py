import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os, glob, time, warnings

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as sps
import pyttsx3  # text-to-speech

# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------
FS_RECORD = 44100          # microphone sample rate
RECORD_SECONDS = 3         # seconds
DATASET_DIR = "Data/"      # folder that contains your .wav files
WORK_FS = 16000            # processing sample rate (downsample for speed)
RECORD_FILE = "record_latest.wav"  # single file reused each recording
SIMILARITY_THRESHOLD = 0.15

# -------------------------------------------------------------------
# STATE
# -------------------------------------------------------------------
recorded_data = None
recorded_path = None
best_match_path = None

# TTS engine (initialized once)
_tts_engine = pyttsx3.init()

def speak_async(text: str):
    """Speak text in a background thread so the UI remains responsive."""
    def _run():
        try:
            _tts_engine.stop()
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)
    threading.Thread(target=_run, daemon=True).start()

# -------------------------------------------------------------------
# AUDIO UTILS
# -------------------------------------------------------------------
def _to_mono_float(x):
    """Ensure 1D float32 array in [-1,1] range."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = x[:, 0]
    # convert to float32 if int
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    return x

def _resample(sig, fs_from, fs_to):
    """Resample using polyphase for quality & speed."""
    if fs_from == fs_to:
        return sig
    g = np.gcd(int(fs_from), int(fs_to))
    up = fs_to // g
    down = fs_from // g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sps.resample_poly(sig, up, down).astype(np.float32)

def _safe_play(wave_, fs):
    try:
        sd.stop()
        sd.play(wave_, fs)
    except Exception as e:
        messagebox.showerror("Audio Error", str(e))

def _load_wav(path, target_fs=None):
    """Read wav -> mono float32, optionally resampled to target_fs."""
    rate, data = wav.read(path)
    data = _to_mono_float(data)
    if target_fs is not None:
        data = _resample(data, rate, target_fs)
        rate = target_fs
    return rate, data

# -------------------------------------------------------------------
# MATCHING LOGIC
# -------------------------------------------------------------------
def normalized_xcorr(a, b):
    """Max normalized cross-correlation between a and b."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    corr = sps.fftconvolve(a, b[::-1], mode="valid")  # slide b over a
    denom = na * nb
    corr = corr / denom
    return float(np.max(corr))

def trim_silence(x, thr=0.02, win=2048):
    """Simple silence trimming by magnitude threshold."""
    if len(x) == 0:
        return x
    x = np.asarray(x, dtype=np.float32)
    eps = 1e-8
    sq = x**2
    kernel = np.ones(win) / win
    rms = np.sqrt(np.convolve(sq, kernel, mode="same") + eps)
    mask = rms > thr
    if not np.any(mask):
        return x
    idx = np.where(mask)[0]
    start, end = int(idx[0]), int(idx[-1]) + 1
    return x[start:end]

def find_best_match(record_path):
    """
    Compare the recorded wav with every .wav in DATASET_DIR.
    Returns (best_path, best_score, person, file_name).
    """
    _, rec = _load_wav(record_path, target_fs=WORK_FS)
    rec = trim_silence(rec)

    best_score, best_path = -1.0, None
    for path in glob.glob(os.path.join(DATASET_DIR, "*.wav")):
        try:
            _, ref = _load_wav(path, target_fs=WORK_FS)
            ref = trim_silence(ref)
            a, b = (rec, ref) if len(rec) >= len(ref) else (ref, rec)
            score = normalized_xcorr(a, b)
            if score > best_score:
                best_score = score
                best_path = path
        except Exception as e:
            print(f"Skipping {path}: {e}")

    person = "Unknown"
    file_name = os.path.basename(best_path) if best_path else "—"
    if " - " in file_name:
        person = file_name.split(" - ")[0]

    return best_path, best_score, person, file_name

# -------------------------------------------------------------------
# BUTTON ACTIONS
# -------------------------------------------------------------------
def start_recording():
    record_btn.config(state="disabled")
    status_var.set("Recording started...")
    threading.Thread(target=_record_flow, daemon=True).start()

def _record_flow():
    global recorded_data, recorded_path, best_match_path
    try:
        frames = int(RECORD_SECONDS * FS_RECORD)
        audio = sd.rec(frames, samplerate=FS_RECORD, channels=1, dtype="float32")
        sd.wait()
        recorded_data = audio.squeeze()

        status_var.set("Finished recording.")
        recorded_path = os.path.abspath(RECORD_FILE)  # overwrite one file every time
        wav.write(recorded_path, FS_RECORD, recorded_data.astype(np.float32))

        status_var.set("Started correlation...")
        t0 = time.time()
        best_match_path, best_score, person, fname = find_best_match(recorded_path)
        dt = time.time() - t0
        status_var.set(f"Finished correlation ({dt:.2f} s)")  # 2 decimals like script.py

        reply_text = ""
        if best_match_path is None or best_score < SIMILARITY_THRESHOLD:
            best_match_var.set("Best Match: (none found)")
        else:
            best_match_var.set(f"Best Match: {person} – {fname} | corr = {best_score:.2f}")

            # -------- phrase-key reply logic (case-insensitive) --------
            phrase_key = os.path.splitext(fname)[0].split(" - ")[-1].strip().lower()
            identified_speaker = person
            identified_phrase = phrase_key

            if phrase_key == "hello":
                response_text = f"Hey! {identified_speaker}"
            elif phrase_key == "how are you":
                response_text = f"I am fine thank you, {identified_speaker}!"
            else:
                response_text = f"{identified_phrase}, {identified_speaker}."
            reply_text = response_text
            # -----------------------------------------------------------

        # Update UI
        reply_var.set(f"Reply: {reply_text}" if reply_text else "Reply: —")

        # Speak every time results appear (i.e., when we have a reply)
        if reply_text:
            speak_async(reply_text)

    except Exception as e:
        messagebox.showerror("Recording Error", str(e))
    finally:
        record_btn.config(state="normal")

def play_recorded():
    if recorded_data is None:
        messagebox.showinfo("Info", "No recording yet.")
        return
    _safe_play(recorded_data, FS_RECORD)

def play_best_match():
    if not best_match_path:
        messagebox.showinfo("Info", "No best match yet.")
        return
    rate, data = _load_wav(best_match_path)  # play at original rate
    _safe_play(data, rate)

# -------------------------------------------------------------------
# THEME
# -------------------------------------------------------------------
BG = "#1e1e1e"
FG = "#ffffff"
SUB = "#dcdcdc"
BTN_BG = "#3a3a3a"
BTN_ACTIVE = "#0078d7"
HEADER_BG = "#0078d7"
HEADER_FG = "#ffffff"
FONT_BASE = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")
FONT_TITLE = ("Segoe UI", 13, "bold")

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
root = tk.Tk()
root.title("Speech-Recognition-Without-AI")
root.geometry("580x580")
root.resizable(False, False)
root.configure(bg=BG)

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", background=BG, foreground=FG, font=FONT_BASE)
style.configure("Secondary.TLabel", background=BG, foreground=SUB, font=FONT_BASE)
style.configure("TButton", background=BTN_BG, foreground=FG, font=FONT_BASE, padding=6)
style.map("TButton", background=[("active", BTN_ACTIVE)])
style.configure("TLabelframe", background=BG, foreground=FG, font=FONT_BOLD)
style.configure("TLabelframe.Label", background=BG, foreground=SUB, font=FONT_BOLD)

header = tk.Frame(root, bg=HEADER_BG, height=44)
header.pack(fill="x")
tk.Label(header, text="Speech Recognition", bg=HEADER_BG, fg=HEADER_FG, font=FONT_TITLE).pack(pady=8)

container = tk.Frame(root, bg=BG)
container.pack(fill="both", expand=True, padx=16, pady=16)

record_btn = ttk.Button(container, text="🎙  Record", command=start_recording)
record_btn.pack(pady=10)

status_frame = ttk.LabelFrame(container, text="Output of the User")
status_frame.pack(fill="x", pady=12)
status_frame.configure(borderwidth=2)
status_var = tk.StringVar(value="Waiting for input...")
ttk.Label(status_frame, textvariable=status_var, wraplength=520).pack(padx=12, pady=10, anchor="w")

answer_frame = ttk.LabelFrame(container, text="Answer")
answer_frame.pack(fill="x", pady=12)
answer_frame.configure(borderwidth=2)
best_match_var = tk.StringVar(value="Best Match: —")
reply_var = tk.StringVar(value="Reply: —")
ttk.Label(answer_frame, textvariable=best_match_var, wraplength=520).pack(padx=12, pady=(10, 4), anchor="w")
ttk.Label(answer_frame, textvariable=reply_var, wraplength=520, style="Secondary.TLabel").pack(padx=12, pady=(0, 10), anchor="w")

audio_block = tk.Frame(container, bg=BG)
audio_block.pack(fill="x", pady=10)

bm_row = tk.Frame(audio_block, bg=BG); bm_row.pack(fill="x", pady=6)
ttk.Label(bm_row, text="Best Match").pack(side="left", padx=(0, 10))
ttk.Button(bm_row, text="Play", command=play_best_match).pack(side="left")

rec_row = tk.Frame(audio_block, bg=BG); rec_row.pack(fill="x", pady=6)
ttk.Label(rec_row, text="Recorded Signal").pack(side="left", padx=(0, 10))
ttk.Button(rec_row, text="Play", command=play_recorded).pack(side="left")



if __name__ == "__main__":
    root.mainloop()
