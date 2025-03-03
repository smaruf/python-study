import re
import json
import os
from gtts import gTTS
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

def extract_words(filename):
    with open(filename, 'r') as file:
        content = file.read()

    pattern = re.compile(r'\d+\.\s\*\*(.*?)\*\*\n\s+-\s\*\*Noun:\*\*\s(.*?)\n\s+-\s\*\*Simple:\*\*\s(.*?)\n\s+-\s\*\*Past:\*\*\s(.*?)\n\s+-\s\*\*Continuous:\*\*\s(.*?)\n\s+-\s\*\*Perfect:\*\*\s(.*?)\n\s+-\s\*\*Pronunciation:\*\*\s(.*?)\n\s+-\s\*\*Active:\*\*\s"(.*?)"\n\s+-\s\*\*Passive:\*\*\s"(.*?)"\n\s+-\s\*\*Question:\*\*\s"(.*?)"', re.DOTALL)
    matches = pattern.findall(content)

    words = []
    for match in matches:
        word = {
            "Word": match[0],
            "Noun": match[1],
            "Simple": match[2],
            "Past": match[3],
            "Continuous": match[4],
            "Perfect": match[5],
            "Pronunciation": match[6],
            "Active": match[7],
            "Passive": match[8],
            "Question": match[9]
        }
        words.append(word)
    return words

def create_flashcards(words):
    flashcards = []
    for word in words:
        card = {
            "word": word["Word"],
            "details": {
                "Noun": word["Noun"],
                "Simple": word["Simple"],
                "Past": word["Past"],
                "Continuous": word["Continuous"],
                "Perfect": word["Perfect"],
                "Pronunciation": word["Pronunciation"],
                "Active": word["Active"],
                "Passive": word["Passive"],
                "Question": word["Question"]
            }
        }
        flashcards.append(card)
    return flashcards

def generate_speech(word, text):
    tts = gTTS(text=text, lang='en')
    filename = f"{word}.mp3"
    tts.save(filename)
    return filename

def on_flashcard_click(word, details):
    msg = f"Word: {word}\n\nNoun: {details['Noun']}\nSimple: {details['Simple']}\nPast: {details['Past']}\nContinuous: {details['Continuous']}\nPerfect: {details['Perfect']}\nPronunciation: {details['Pronunciation']}\nActive: {details['Active']}\nPassive: {details['Passive']}\nQuestion: {details['Question']}"
    messagebox.showinfo("Word Details", msg)
    filename = generate_speech(word, word)
    playsound(filename)
    os.remove(filename)

def create_gui(flashcards):
    root = tk.Tk()
    root.title("Flashcards")

    for card in flashcards:
        button = tk.Button(root, text=card["word"], command=lambda c=card: on_flashcard_click(c["word"], c["details"]))
        button.pack(pady=5)

    root.mainloop()

def main():
    filename = 'English_c1_words.md'
    words = extract_words(filename)
    flashcards = create_flashcards(words)
    create_gui(flashcards)

if __name__ == "__main__":
    main()
