import tkinter as tk
from tkinter import simpledialog
import json
import random
from datetime import datetime
import pyttsx3
import os

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("C1 English Proficiency App")

        self.points = 0
        self.level = 1
        self.username = "User"
        self.leaderboard = {}
        self.flashcards = self.load_flashcards()

        self.engine = pyttsx3.init()
        
        self.label = tk.Label(master, text="Welcome to the C1 English Proficiency Flashcard App!")
        self.label.pack()

        self.points_label = tk.Label(master, text=f"Points: {self.points} Level: {self.level}")
        self.points_label.pack()

        self.next_button = tk.Button(master, text="Next Flashcard", command=self.present_next_flashcard)
        self.next_button.pack()

    def load_flashcards(self):
        if not os.path.exists('flashcards.json'):
            return []
        with open('flashcards.json', 'r') as file:
            return json.load(file)

    def save_flashcards(self):
        with open('flashcards.json', 'w') as file:
            json.dump(self.flashcards, file, indent=4)

    def update_points(self, correct):
        if correct:
            self.points += 10
            self.check_level_up()
        self.update_points_label()

    def update_points_label(self):
        self.points_label.config(text=f"Points: {self.points} Level: {self.level}")

    def check_level_up(self):
        required_points = self.level * 100  # Example scaling
        if self.points >= required_points:
            self.level += 1
            self.level_up_notification()

    def level_up_notification(self):
        print(f"Congratulations! You've reached Level {self.level}!")

    def present_next_flashcard(self):
        if not self.flashcards:
            self.label.config(text="No flashcards available. Please add some!")
            return

        flashcard = random.choice(self.flashcards)
        answer = simpledialog.askstring("Question", flashcard['question'])
        
        if answer is None:  # User closed the dialog
            return

        if answer.lower().strip() == flashcard['answer'].lower().strip():
            self.update_points(True)
            self.label.config(text="Correct!")
        else:
            self.label.config(text=f"Wrong. The correct answer is: {flashcard['answer']}")

    def add_flashcard(self, category, question, answer):
        flashcard = {"category": category, "question": question, "answer": answer}
        self.flashcards.append(flashcard)
        self.save_flashcards()
        print("New flashcard added successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()
