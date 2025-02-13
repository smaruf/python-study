import tkinter as tk
import random

# Flashcards data: word as key and hint as value
flashcards = {
    "Employability": "Ability to gain and maintain a job",
    "Expertise": "Advanced knowledge or skill",
    "Collocations": "Words that are often used together",
    "Performance": "How well a task is executed",
    "Feedback": "Information provided regarding one's performance",
    "Appraisals": "Evaluations of an employee's performance",
    "Constructive": "Serving a useful purpose",
    "Probability": "Likelihood of something happening",
    "Future perfect": "Grammar tense indicating an action will be finished",
    "Future continuous": "Grammar tense indicating an ongoing action",
    "Developments": "Progress or evolution of something",
    "Predictions": "Statements about what will happen",
    "Monitor": "To observe or check",
    "Criteria": "Standards by which something is judged",
    "Rate objectives": "Evaluate goals or targets",
    "Express views": "State one's opinions",
    "Address concerns": "Deal with worries or issues",
    "Raise issues": "Bring problems to attention",
    "Respond to criticism": "Reply to disapproval",
    "Pronunciation": "The way in which a word is pronounced",
    "Vocabulary": "Collection of words used",
    "Grammar": "Rules of language composition",
    "Revision": "Act of altering or reconsidering",
    "Practice": "Repeated activities to improve skills",
    "Achieve": "Successfully reach an objective",
    "Discuss": "Talk about something"
}

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("Performance Vocabulary Flashcards")

        self.label = tk.Label(master, text="", font=('Helvetica', 14), height=4)
        self.label.pack()

        self.hint_entry = tk.Entry(master)
        self.hint_entry.pack()

        self.check_button = tk.Button(master, text="Check", command=self.check_hint)
        self.check_button.pack()

        self.next_button = tk.Button(master, text="Next", command=self.next_flashcard)
        self.next_button.pack()
        
        self.result_label = tk.Label(master, text="", font=('Helvetica', 14))
        self.result_label.pack()

        self.next_flashcard()

    def next_flashcard(self):
        self.current_word, self.current_hint = random.choice(list(flashcards.items()))
        self.label.config(text=f"What is the hint for: {self.current_word}?")
        self.hint_entry.delete(0, tk.END)
        self.result_label.config(text="")

    def check_hint(self):
        user_input = self.hint_entry.get()
        if user_input.lower() == self.current_hint.lower():
            self.result_label.config(text="Correct!", fg="green")
        else:
            self.result_label.config(text=f"Wrong! Correct hint: {self.current_hint}", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()
