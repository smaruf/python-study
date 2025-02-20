import tkinter as tk
import random
from tkinter import messagebox, simpledialog
import pyttsx3
import json
import os
from datetime import datetime, timedelta
import wikipediaapi
import openai

# Import flashcards from flashcards.py
from src.learning_english.flashcards import flashcards as base_flashcards

class CombinedFlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("Combined Flashcard App")

        # Initialize base components
        self.points = 0
        self.level = 1
        self.username = "User"
        self.leaderboard = {}
        self.flashcards = self.load_flashcards()
        self.personalized_learning_path = []

        self.engine = pyttsx3.init()
        
        self.label = tk.Label(master, text="Welcome to the Combined Flashcard App!")
        self.label.pack()

        self.points_label = tk.Label(master, text=f"Points: {self.points} Level: {self.level}")
        self.points_label.pack()
        
        self.correct_answers = 0
        self.attempted_questions = 0
        
        self.question_frame = tk.Frame(master)
        self.question_frame.pack(pady=20)
        
        self.display_text = tk.Label(self.question_frame, text="", font=('Helvetica', 14), wraplength=400)
        self.display_text.pack(side=tk.TOP)
        
        self.options_frame = tk.Frame(self.question_frame)
        self.var = tk.StringVar()
        
        self.user_input = tk.Entry(self.question_frame, font=('Helvetica', 14), width=50)
        
        self.check_button = tk.Button(master, text="Check", command=self.check_answer)
        self.check_button.pack(pady=10)
        
        self.feedback_label = tk.Label(master, text="", font=('Helvetica', 14))
        self.feedback_label.pack()
        
        self.next_button = tk.Button(master, text="Next", command=self.present_next_flashcard)
        self.next_button.pack(pady=20)
        
        self.next_flashcard()

    def load_flashcards(self):
        if not os.path.exists('flashcards.json'):
            return base_flashcards
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
        messagebox.showinfo("Level Up", f"Congratulations! You've reached Level {self.level}!")

    def present_next_flashcard(self):
        if not self.flashcards:
            self.label.config(text="No flashcards available. Please add some!")
            return

        # Select flashcards due for review based on spaced repetition
        due_flashcards = [fc for fc in self.flashcards if datetime.fromisoformat(fc.get('next_review', '1970-01-01')) <= datetime.now()]
        if not due_flashcards:
            self.label.config(text="No flashcards due for review. Please wait or add more flashcards!")
            return

        self.current_card = random.choice(due_flashcards)
        self.label.config(text=f"Question: {self.current_card['question']}")
        self.user_input.delete(0, tk.END)
        self.options_frame.pack_forget()
        
        if self.current_card.get("type") == "text":
            self.user_input.pack()
        elif self.current_card.get("type") == "mcq":
            self.user_input.pack_forget()
            self.options_frame.pack()
            for widget in self.options_frame.winfo_children():
                widget.destroy()
            for option in self.current_card.get("options", []):
                tk.Radiobutton(self.options_frame, text=option, variable=self.var, value=option, font=('Helvetica', 12)).pack(anchor="w")

        self.feedback_label.config(text="")
        self.next_button.config(state="disabled")

    def check_answer(self):
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
        self.present_next_flashcard()

    def pronounce_text(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

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
    app = CombinedFlashcardApp(root)
    root.mainloop()
