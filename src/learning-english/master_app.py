import tkinter as tk
import random
from tkinter import messagebox, simpledialog
import pyttsx3
import json
import os
from datetime import datetime, timedelta
import wikipediaapi
import openai
from cryptography.fernet import Fernet

# Import flashcards from flashcards.py
from src.learning_english.flashcards import flashcards as base_flashcards

class CombinedFlashcardApp:
    """A class to represent the Combined Flashcard Application."""

    def __init__(self, master):
        """
        Initialize the Combined Flashcard Application.

        :param master: The master widget.
        """
        self.master = master
        master.title("Combined Flashcard App")
        self.initialize_components()
        self.setup_ui()
        self.api_key = self.get_api_key()

    def initialize_components(self):
        """Initialize base components for the application."""
        self.points = 0
        self.level = 1
        self.username = "User"
        self.leaderboard = {}
        self.flashcards = self.load_flashcards()
        self.personalized_learning_path = []
        self.engine = pyttsx3.init()
        self.correct_answers = 0
        self.attempted_questions = 0
        self.var = tk.StringVar()

    def setup_ui(self):
        """Set up the user interface."""
        self.label = tk.Label(self.master, text="Welcome to the Combined Flashcard App!")
        self.label.pack()

        self.points_label = tk.Label(self.master, text=f"Points: {self.points} Level: {self.level}")
        self.points_label.pack()

        self.question_frame = tk.Frame(self.master)
        self.question_frame.pack(pady=20)

        self.display_text = tk.Label(self.question_frame, text="", font=('Helvetica', 14), wraplength=400)
        self.display_text.pack(side=tk.TOP)

        self.options_frame = tk.Frame(self.question_frame)
        self.user_input = tk.Entry(self.question_frame, font=('Helvetica', 14), width=50)

        self.check_button = tk.Button(self.master, text="Check", command=self.check_answer)
        self.check_button.pack(pady=10)

        self.feedback_label = tk.Label(self.master, text="", font=('Helvetica', 14))
        self.feedback_label.pack()

        self.next_button = tk.Button(self.master, text="Next", command=self.present_next_flashcard)
        self.next_button.pack(pady=20)

        self.next_flashcard()

    def get_api_key(self):
        """Prompt the user for the OpenAI API key and store it encrypted in a JSON file."""
        key_file = 'api_key.json'
        if os.path.exists(key_file):
            with open(key_file, 'r') as file:
                data = json.load(file)
                fernet = Fernet(data['key'])
                return fernet.decrypt(data['api_key'].encode()).decode()

        api_key = simpledialog.askstring("Input", "Please enter your OpenAI API key:", show='*')
        encryption_key = Fernet.generate_key()
        fernet = Fernet(encryption_key)
        encrypted_api_key = fernet.encrypt(api_key.encode()).decode()

        with open(key_file, 'w') as file:
            json.dump({'key': encryption_key.decode(), 'api_key': encrypted_api_key}, file)

        return api_key

    def load_flashcards(self):
        """
        Load flashcards from a JSON file or use base flashcards.

        :return: List of flashcards.
        """
        if not os.path.exists('flashcards.json'):
            return base_flashcards
        try:
            with open('flashcards.json', 'r') as file:
                return json.load(file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load flashcards: {e}")
            return base_flashcards

    def save_flashcards(self):
        """Save flashcards to a JSON file."""
        try:
            with open('flashcards.json', 'w') as file:
                json.dump(self.flashcards, file, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save flashcards: {e}")

    def update_points(self, correct, flashcard):
        """
        Update points and review times based on the answer correctness.

        :param correct: Boolean indicating if the answer was correct.
        :param flashcard: The current flashcard being reviewed.
        """
        if correct:
            self.points += 10
            flashcard['next_review'] = (datetime.now() + timedelta(days=1)).isoformat()  # Example of spaced repetition
            self.check_level_up()
        else:
            flashcard['next_review'] = (datetime.now() + timedelta(hours=1)).isoformat()  # Review sooner if incorrect
        self.update_points_label()
        self.save_flashcards()

    def update_points_label(self):
        """Update the points and level label."""
        self.points_label.config(text=f"Points: {self.points} Level: {self.level}")

    def check_level_up(self):
        """Check if the user qualifies for a level up."""
        required_points = self.level * 100  # Example scaling
        if self.points >= required_points:
            self.level += 1
            self.level_up_notification()

    def level_up_notification(self):
        """Notify the user about level up."""
        messagebox.showinfo("Level Up", f"Congratulations! You've reached Level {self.level}!")

    def present_next_flashcard(self):
        """Present the next flashcard to the user."""
        if not self.flashcards:
            self.label.config(text="No flashcards available. Please add some!")
            return

        # Select flashcards due for review based on spaced repetition
        due_flashcards = [fc for fc in self.flashcards if datetime.fromisoformat(fc.get('next_review', '1970-01-01')) <= datetime.now()]
        if not due_flashcards:
            self.label.config(text="No flashcards due for review. Please wait or add more flashcards!")
            return

        self.current_card = random.choice(due_flashcards)
        self.display_flashcard()

    def display_flashcard(self):
        """Display the current flashcard question and options."""
        self.label.config(text=f"Question: {self.current_card['question']}")
        self.user_input.delete(0, tk.END)
        self.options_frame.pack_forget()

        if self.current_card.get("type") == "text":
            self.user_input.pack()
        elif self.current_card.get("type") == "mcq":
            self.user_input.pack_forget()
            self.options_frame.pack()
            self.populate_mcq_options()

        self.feedback_label.config(text="")
        self.next_button.config(state="disabled")

    def populate_mcq_options(self):
        """Populate multiple choice question options."""
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        for option in self.current_card.get("options", []):
            tk.Radiobutton(self.options_frame, text=option, variable=self.var, value=option, font=('Helvetica', 12)).pack(anchor="w")

    def check_answer(self):
        """Check the user's answer and provide feedback."""
        if self.current_card.get("type") == "text":
            user_input = self.user_input.get()
            if user_input.lower().strip() == self.current_card.get("answer", "").lower().strip():
                self.feedback_label.config(text="Correct!", fg="green")
                self.correct_answers += 1
            else:
                self.feedback_label.config(text=f"Wrong! The correct answer is: {self.current_card.get('answer')}", fg="red")
        elif self.current_card.get("type") == "mcq":
            if self.var.get() == self.current_card.get("answer"):
                self.feedback_label.config(text="Correct!", fg="green")
                self.correct_answers += 1
            else:
                self.feedback_label.config(text=f"Wrong! The correct answer is: {self.current_card.get('answer')}", fg="red")

        self.attempted_questions += 1
        self.next_button.config(state="normal")

    def next_flashcard(self):
        """Proceed to the next flashcard."""
        self.present_next_flashcard()

    def pronounce_text(self, text):
        """
        Use text-to-speech to pronounce given text.

        :param text: Text to be pronounced.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def add_flashcard(self, category, question, answer):
        """
        Add a new flashcard to the set.

        :param category: The category of the flashcard.
        :param question: The question of the flashcard.
        :param answer: The answer of the flashcard.
        """
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
        """
        Fetch word difficulty using Wikipedia API.

        :param word: The word to fetch difficulty for.
        :return: The difficulty summary.
        """
        wiki_wiki = wikipediaapi.Wikipedia('en')
        page = wiki_wiki.page(word)
        if page.exists():
            summary = page.summary[:60]  # Get the first 60 characters
            return summary
        return 'unknown'

    def get_chatgpt_response(self, prompt):
        """
        Fetch a response from ChatGPT.

        :param prompt: The prompt to send to ChatGPT.
        :return: The response from ChatGPT.
        """
        openai.api_key = self.api_key
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        return response.choices[0].text.strip()

    def fetch_flashcard_info(self, word):
        """
        Fetch flashcard information from Wikipedia and ChatGPT.

        :param word: The word to fetch information for.
        :return: Combined information from Wikipedia and ChatGPT.
        """
        wiki_info = self.get_word_difficulty(word)
        chatgpt_info = self.get_chatgpt_response(f"Provide a summary for the word '{word}'")
        return f"Wikipedia: {wiki_info}\nChatGPT: {chatgpt_info}"

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedFlashcardApp(root)
    root.mainloop()
