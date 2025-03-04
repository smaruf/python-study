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

    pattern = re.compile(r'\\d+\\.\\s\\*\\*(.*?)\\*\\*\\n\\s+-\\s\\*\\*Noun:\\*\\*\\s(.*?)\\n\\s+-\\s\\*\\*Forms:\\*\\*\\s(.*?)\\n\\s+-\\s\\*\\*Pronunciation:\\*\\*\\s(.*?)\\n\\s+-\\s\\*\\*Examples:\\*\\*\\s(.*?)\\n')
    matches = pattern.findall(content)

    words = []
    for match in matches:
        word = {
            "Word": match[0],
            "Noun": match[1],
            "Forms": match[2],
            "Pronunciation": match[3],
            "Examples": match[4]
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
                "Forms": word["Forms"],
                "Pronunciation": word["Pronunciation"],
                "Examples": word["Examples"],
                "Definition": get_meaning(word["Word"])
            }
        }
        flashcards.append(card)
    return flashcards

def get_meaning(word):
    dictionary = {
        "Augment": "To make something greater by adding to it; increase. Synonym: Enhance, Amplify",
        "Facilitate": "Make (an action or process) easy or easier. Synonym: Ease, Enable"
        # Add more words and their meanings here...
    }
    return dictionary.get(word, "Meaning not found")

def generate_speech(word, text):
    tts = gTTS(text=text, lang='en')
    filename = f"{word}.mp3"
    tts.save(filename)
    return filename

def on_flashcard_click(word, details):
    msg = f"Word: {word}\n\nNoun: {details['Noun']}\nForms: {details['Forms']}\nPronunciation: {details['Pronunciation']}\nExamples: {details['Examples']}\nDefinition: {details['Definition']}"
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
