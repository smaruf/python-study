import tkinter as tk
from tkinter import simpledialog, messagebox, Frame, Label, Button
from gtts import gTTS
from io import BytesIO
import os
import openai
import json
import playsound
import threading

# Configure API Key here
openai_api_key = 'your-openai-api-key'

# Load JSON Data
def load_data(filename="vocab_data.json"):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}
    return data

# Generate and Play Pronunciation
def play_pronunciation(text):
    tts = gTTS(text)
    # Using BytesIO to avoid saving audio files
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    # Play sound on a separate thread to prevent the GUI from freezing
    threading.Thread(target=lambda: playsound.playsound(fp)).start()

# OpenAI Query
def ask_question(question, word_details):
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Use this data to answer the question: {word_details}\nQuestion: {question}",
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# GUI Setup
def setup_gui(words):
    root = tk.Tk()
    root.title("Advanced Vocabulary Assistant")

    main_frame = Frame(root)
    main_frame.pack(pady=20)

    word_label = Label(main_frame, text="Enter the word you want to learn about:")
    word_label.pack()

    word_entry = tk.Entry(main_frame)
    word_entry.pack()

    def on_ask():
        word = word_entry.get()
        if word in words:
            question = simpledialog.askstring(
                "Input", "What's your question?",
                parent=root
            )
            if question:
                answer = ask_question(question, words[word])
                messagebox.showinfo("Answer", answer)
                play_pronunciation(words[word]['pronunciation'])
            else:
                messagebox.showinfo("Note", "No question asked!")
        else:
            messagebox.showwarning("Error", "Word not found!")
    
    ask_button = Button(main_frame, text="Ask About Word", command=on_ask)
    ask_button.pack()

    root.mainloop()

# Main function
if __name__ == "__main__":
    word_data = load_data()
    setup_gui(word_data)
