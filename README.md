\# 🗣️ Speech Recognition Without AI (Cross-Correlation Based)



This project implements a \*\*non-AI voice recognition system\*\* that identifies \*who\* is speaking and \*what phrase\* is said using \*\*signal cross-correlation\*\* — no neural networks or machine learning required.



It compares a live-recorded voice clip with preprocessed audio samples and responds both on-screen and through synthesized speech.



---



\## 🎯 Features



\- 🎙️ Records short live audio (default 3 seconds)  

\- 🧹 Preprocesses all dataset WAVs (normalization + noise gating)  

\- ⚖️ Matches audio by \*\*normalized cross-correlation\*\*  

\- 🧍 Identifies both \*\*speaker\*\* and \*\*phrase\*\*  

\- 🕒 Displays \*\*total correlation time\*\* and best match file name  

\- 🗣️ Responds with \*\*text-to-speech\*\* feedback  



---



\## 📂 Project Structure



Speech-Recognition-without-AI/

│

├── Data/ # Original recorded WAV dataset

│ ├── Mohammad - Hello.wav

│ ├── Mohammad - Hello2.wav

│ ├── Mohammad - How are you.wav

│ └── ... etc.

│

├── Preprocessed/ # Auto-generated cleaned WAVs (output)

│

├── dataset\_processing.py # Cleans and normalizes all dataset WAVs

├── script.py # Main recognition logic

├── requirements.txt # Dependencies

└── README.md



---



\## ⚙️ Requirements



Install dependencies using:


pip install numpy sounddevice pyttsx3



> Optional: use `ffmpeg` for audio format conversion if needed.



Also ensure you have a \*\*working microphone\*\* connected to your computer.



---



\## 🧩 Preparing the Dataset



1\. Place all audio recordings inside the `Data/` folder.  

&nbsp;  Each filename must follow this pattern:


{SpeakerName} - {Phrase}.wav



Examples:

Malek - Hello.wav

Manar - How are you.wav

Mohammad - Hello2.wav



> If a filename ends with a number (e.g., `Hello2.wav`), the system automatically ignores the numeric suffix when interpreting the phrase.



2\. Run preprocessing:
python dataset\_processing.py
This cleans all `.wav` files and saves them to the `Preprocessed/` directory.



---



\## ▶️ Running the Recognition System



Once preprocessing is done:


python script.py



You’ll see output like this:


Listening for an audio phrase...

Finished Listening, started processing...

Response: Hey! Mohammad

Max correlation coefficient: 0.186

Best match file: Mohammad - Hello2.wav

Total correlation time: 47.12 s



The system will also \*\*speak the response aloud\*\*.



---



\## 🧠 How It Works



\### 🔊 Preprocessing

\- Converts all signals to mono, normalized amplitude  

\- Removes DC offset  

\- Applies a noise gate to eliminate low-level background noise  

\- Saves cleaned signals for faster matching



\### ⚡ Matching

\- Computes \*\*maximum normalized cross-correlation\*\* between the recorded signal and each reference sample.  

\- Tracks:

&nbsp; - The speaker \& phrase with the highest match

&nbsp; - The correlation coefficient

&nbsp; - The total processing time



\### 🗣️ Response

\- If correlation ≥ `SIMILARITY\_THRESHOLD`, a spoken and printed message is produced:

&nbsp; - `"Hello"` → `"Hey! {speaker}"`

&nbsp; - `"How are you"` → `"I am fine thank you, {speaker}!"`



---



\## 🔬 Example Output

Listening for an audio phrase...

Finished Listening, started processing...

Response: I am fine thank you, Mohammad!

Max correlation coefficient: 0.223

Best match file: Mohammad - How are you3.wav

Total correlation time: 39.8 s



---



\## 🚀 Future Enhancements



\- 📈 Add dynamic thresholding or confidence scaling  

\- 🎧 Automatic silence trimming  

\- 🧩 Integration with a lightweight GUI  

\- 🌐 Add support for longer phrases or continuous recognition  



---



\### 🏁 Author \& Contributors

\- \*\*Project Lead:\*\* Mohammad  

\- \*\*Collaborators:\*\* Manar, Malek, Oussama, Hadi  



---



✅ The project is now fully functional and optimized for clear, fast cross-correlation–based voice recognition.















