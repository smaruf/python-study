import tkinter as tk
from tkinter import messagebox
import pyttsx3

# Import flashcards from flashcards.py
from src.learning_english.flashcards import flashcards

# Function to initialize text-to-speech engine
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    return engine

# Function to pronounce text
def pronounce_text(text):
    engine = init_tts()
    engine.say(text)
    engine.runAndWait()

# Function to display the next flashcard
def next_flashcard():
    global current_card_index
    if current_card_index < len(flashcards):
        card = flashcards[current_card_index]
        question_label.config(text=f"Category: {card['category']}\nQuestion: {card['question']}")
        answer_entry.delete(0, tk.END)
    else:
        messagebox.showinfo("End", "You have completed all flashcards.")
        root.quit()

# Function to check the answer
def check_answer():
    global score
    card = flashcards[current_card_index]
    user_answer = answer_entry.get().strip().lower()
    correct_answer = card['answer'].strip().lower()
    if user_answer == correct_answer:
        score += 1
        messagebox.showinfo("Correct", "Correct!")
    else:
        messagebox.showinfo("Incorrect", f"Incorrect. The correct answer is: {card['answer']}")
    current_card_index += 1
    next_flashcard()

# Function to pronounce the answer
def pronounce_answer():
    card = flashcards[current_card_index]
    pronounce_text(card['answer'])

# Initialize GUI
root = tk.Tk()
root.title("Flashcard App")

current_card_index = 0
score = 0

question_label = tk.Label(root, text="", wraplength=400)
question_label.pack(pady=20)

answer_entry = tk.Entry(root, width=50)
answer_entry.pack(pady=10)

check_button = tk.Button(root, text="Check Answer", command=check_answer)
check_button.pack(pady=5)

pronounce_button = tk.Button(root, text="Pronounce Answer", command=pronounce_answer)
pronounce_button.pack(pady=5)

next_flashcard()

root.mainloop()
