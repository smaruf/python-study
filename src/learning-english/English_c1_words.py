import re
import os
from gtts import gTTS
import tkinter as tk
from tkinter import messagebox
from playsound import playsound

def parse_md_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    pattern = re.compile(r"### (\w+)\n- \*\*Pronunciation:\*\* (.*?)\n- \*\*Definition:\*\* (.*?)\n- \*\*Synonym:\*\* (.*?)\n")
    matches = pattern.findall(content)

    vocab_dict = {match[0]: {
                    'Pronunciation': match[1],
                    'Definition': match[2],
                    'Synonym': match[3].split(', ')
                  } for match in matches}

    return vocab_dict

def extract_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    pattern = re.compile(r"\*\*(.*?)\*\*")
    matches = pattern.findall(content)
    
    words = [match.strip() for match in matches]
    return words

def create_flashcards(words, dictionary):
    flashcards = []
    for word in words:
        if word in dictionary:
            card = {
                "word": word,
                "details": dictionary[word]
            }
            flashcards.append(card)
    return flashcards

def generate_speech(word, text):
    tts = gTTS(text=text, lang='en')
    filename = f"{word}.mp3"
    tts.save(filename)
    return filename

def on_flashcard_click(word, details):
    msg = f"Word: {word}\n\nPronunciation: {details['Pronunciation']}\nDefinition: {details['Definition']}\nSynonym: {', '.join(details['Synonym'])}"
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
    words_file = 'src/learning-english/English_c1_words.md'
    dict_file = 'src/learning-english/English_c1_dictionary.md'

    words = extract_words(words_file)
    dictionary = parse_md_to_dict(dict_file)

    flashcards = create_flashcards(words, dictionary)
    create_gui(flashcards)

if __name__ == "__main__":
    main()
