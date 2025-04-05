"""
Arabic Markaz Rules

This module provides an overview of the Arabic language concepts of Harf (letters) and Markaz (positions),
which are essential for natural language processing (NLP) applications. The guide includes examples of
letters in different positions and their usage in NLP tasks such as text normalization, tokenization, 
and morphological analysis.

Requirements:
- Python 3.x
- Tkinter: For creating the GUI.
- SpeechRecognition: For capturing and processing audio input.
- Pydub: For audio processing (optional, if needed for advanced features).
- Transformers: For using advanced NLP models.
- Torch: For using the Wav2Vec2 model.

Installation:
`pip install tk speechrecognition pydub numpy transformers torch`

Usage:
- Run the script to open a GUI that allows the user to practice Arabic pronunciation.
- Click on a letter to start practicing. The application will prompt the user to pronounce the letter,
  capture the audio, and provide feedback on the pronunciation.

Example: following script
"""

import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import numpy as np

class ArabicPronunciationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic Pronunciation Practice")

        self.letters = [
            "أ", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"
        ]
        self.create_widgets()

        # Load Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    def create_widgets(self):
        tk.Label(self.root, text="Click on a letter to practice pronunciation:").pack(pady=10)
        
        for letter in self.letters:
            button = tk.Button(self.root, text=letter, font=("Arial", 24), command=lambda l=letter: self.practice_pronunciation(l))
            button.pack(side=tk.LEFT, padx=5, pady=5)

    def practice_pronunciation(self, letter):
        messagebox.showinfo("Practice", f"Please pronounce the letter: {letter}")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio_data = recognizer.listen(source)
            try:
                # Get audio data as numpy array
                audio_np = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32) / 32768.0

                # Process audio data
                input_values = self.processor(audio_np, return_tensors="pt", sampling_rate=16000).input_values
                logits = self.model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.decode(predicted_ids[0])

                if transcription == letter:
                    messagebox.showinfo("Result", "Correct pronunciation!")
                else:
                    messagebox.showinfo("Result", f"Incorrect pronunciation. You said: {transcription}")
            except sr.UnknownValueError:
                messagebox.showinfo("Error", "Could not understand the audio")
            except sr.RequestError as e:
                messagebox.showinfo("Error", f"Could not request results; {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ArabicPronunciationApp(root)
    root.mainloop()
