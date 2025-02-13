import tkinter as tk
from tkinter import simpledialog
import json
import random
import requests  # For accessing APIs
from datetime import datetime, timedelta
import pyttsx3
import os
import wikipediaapi  # For Wikipedia API
import openai  # For ChatGPT API

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("C1 English Proficiency App")

        self.points = 0
        self.level = 1
        self.username = "User"
        self.leaderboard = {}
        self.flashcards = self.load_flashcards()
        self.personalized_learning_path = []

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

    def update_points(self, correct, flashcard):
        if correct:
            self.points += 10
            flashcard['next_review'] = (datetime.now() + timedelta(days=1)).isoformat()  # Example of spaced repetition
            self.check_level_up()
        else:
            flashcard['next_review'] = (datetime.now() + timedelta(hours=1)).isoformat()  # Review sooner if incorrect
        self.update_points_label()
        self.save_flashcards()

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

        # Select flashcards due for review based on spaced repetition
        due_flashcards = [fc for fc in self.flashcards if datetime.fromisoformat(fc.get('next_review', '1970-01-01')) <= datetime.now()]
        if not due_flashcards:
            self.label.config(text="No flashcards due for review. Please wait or add more flashcards!")
            return

        flashcard = random.choice(due_flashcards)
        answer = simpledialog.askstring("Question", flashcard['question'])
        
        if answer is None:  # User closed the dialog
            return

        if answer.lower().strip() == flashcard['answer'].lower().strip():
            self.update_points(True, flashcard)
            self.label.config(text="Correct!")
        else:
            self.update_points(False, flashcard)
            self.label.config(text=f"Wrong. The correct answer is: {flashcard['answer']}")

    def add_flashcard(self, category, question, answer):
        difficulty = self.get_word_difficulty(answer)
        flashcard = {
            "category": category,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "next_review": datetime.now().isoformat()  # Initial review time
        }
        self.flashcards.append(flashcard)
        self.save_flashcards()
        print("New flashcard added successfully.")

    def get_word_difficulty(self, word):
        # Using Wikipedia API to fetch word difficulty
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(word)
        if page.exists():
            summary = page.summary[:60]  # Get the first 60 characters
            return summary
        return 'unknown'

    def get_chatgpt_response(self, prompt):
        openai.api_key = 'your_openai_api_key'  # Replace with your OpenAI API key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        return response.choices[0].text.strip()

    def fetch_flashcard_info(self, word):
        wiki_info = self.get_word_difficulty(word)
        chatgpt_info = self.get_chatgpt_response(f"Provide a summary for the word '{word}'")
        return f"Wikipedia: {wiki_info}\nChatGPT: {chatgpt_info}"

if __name__ == "__main__":
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()
