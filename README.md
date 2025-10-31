## 🗣️ Speech Recognition Without AI (Cross-Correlation Based)



This project implements a **non-AI voice recognition system** that identifies *who* is speaking and *what phrase* is said using **signal cross-correlation** — no neural networks or machine learning required.



It compares a live-recorded voice clip with preprocessed audio samples and responds both on-screen and through synthesized speech.



---



## 🎯 Features



- 🎙️ Records short live audio 
- 🧹 Preprocesses all dataset WAVs (normalization + noise gating)  
- ⚖️ Matches audio by **normalized cross-correlation**  
- 🧍 Identifies both **speaker** and **phrase**  
- 🕒 Displays **total correlation time** and best match file name  
- 🗣️ Responds with **text-to-speech** based on the identified phrase
- ⚠️ A threshold is set for identification in case no phrase/speaker matched the live recording  
- 🖥️ In the GUI, both the recorded `.wav` file and the recognized `.wav` file are playable


---



## 📂 Project Structure



Speech-Recognition-without-AI/

│

├── Data/ # Original recorded WAV dataset

│ ├── Mohammad - Hello.wav

│ ├── Mohammad - Hello2.wav

│ ├── Mohammad - How are you.wav

│ └── ... etc.

│

├── Preprocessed/ # Auto-generated cleaned WAVs

│

├── dataset\_processing.py # Cleans and normalizes all dataset WAVs

├── script.py # Main recognition logic

├── gui.py # Simple user interface

├── record_latest.wav # latest recorded phrase by the GUI

└── README.md # This file



---



## ⚙️ Requirements



Install dependencies using:

```bash
pip install numpy sounddevice pyttsx3 scipy pillow
```


> Optional: Add code to use `ffmpeg` for audio format conversion if needed.



Also ensure you have a **working microphone** connected to your computer.



---



## 🧩 Preparing the Dataset



1\. Place all audio recordings inside the `Data/` folder.  

&nbsp;  Each filename must follow this pattern:

```bash
{Person} - {Phrase}.wav
```


Examples:
```bash
Malek - Hello.wav

Manar - How are you.wav

Mohammad - Hello2.wav
```


> If a filename ends with a number (e.g., `Hello2.wav`), the system automatically ignores the numeric suffix when interpreting the phrase.
> This is used to store multiple `.wav` files per phrase per person (more dataset variety and thus better recognition)



2\. Run preprocessing:
```bash
python dataset\_processing.py
```

This cleans all `.wav` files and saves them to the `Preprocessed/` directory.



---



## ▶️ Running the Recognition System



Once preprocessing is done:

### Either run:

```bash
python script.py
```

#### For the terminal application


### Or run:

```bash
python gui.py
```

#### for the same logic implemented through a simple interface

You’ll see output like this:

```bash
Listening for an audio phrase...

Finished Listening, started processing...

Response: Hey! Mohammad

Max correlation coefficient: 0.186

Best match file: Mohammad - Hello2.wav

Total correlation time: 47.12 s
```


The system will also **respond out loud**.



---



## 🧠 How It Works

### 🔊 Preprocessing
- Converts all signals to mono, normalized amplitude  
- Removes DC offset  
- Applies a noise gate to eliminate low-level background noise  
- Saves cleaned signals for faster matching


### 🖥️ Graphical User Interface (GUI)

The project features a Tkinter-based GUI that allows users to record speech, compare it with stored samples, and view results interactively.

#### 🎙 Record

Click “Record” to capture audio using your microphone.

The app displays progress messages like “Recording started…” and “Finished correlation (1.5 s)”.

#### 💬 Output of the User

Shows the current status of recording and processing in real time.

#### 🤖 Answer

Displays:

Best Match: the detected speaker and phrase.

Reply: a short text (and TTS) response based on the recognized phrase.

#### 🎧 Playback

Best Match → Play: plays the closest-matching dataset sample.

Recorded Signal → Play: replays your recorded voice for comparison.

#### 🎨 Design

Clean dark theme built with Tkinter + ttk.

Lightweight, responsive, and easy to run on any system.



### 🗣️ Responses

- If correlation ≥ `SIMILARITY\_THRESHOLD`, a spoken and printed message is produced (currently we've only coded 2 responses but you can add as many as you want):

&nbsp; - `"Hello"` → `"Hey! {speaker}"`

&nbsp; - `"How are you"` → `"I am fine thank you, {speaker}!"`



---

### ☕ Adding More Responses
To Add more Responses, simply add more `elif` statements to this part of the code `script.py (lines 117-122)`:
```python
if phrase_key == "hello":
    response_text = f"Hey! {identified_speaker}"
elif phrase_key == "how are you":
    response_text = f"I am fine thank you, {identified_speaker}!"
else:
    response_text = f"You said: {identified_phrase}, {identified_speaker}."
``` 


## 🚀 Future Enhancements

- 📈 Add dynamic thresholding or confidence scaling  
- 🎧 Automatic silence trimming  
- 🌐 Add support for longer phrases or continuous recognition  
- 🔊 Add .wav files for responses instead of simple TTS
