import tkinter as tk
import random

# Expanded flashcards data
flashcards = [
    {"topic": "Negotiating with colleagues", "question": "Discuss areas that need attention in open options.", "type": "text"},
    {"topic": "Handling phone calls", "question": "Manage information flow during calls with clarity.", "type": "text"},
    {"topic": "Talking about changes", "question": "What verb best describes resisting changes?", "type": "mcq", "options": ["resist", "oppose", "affect", "revert"], "answer": "resist"},
    {"topic": "Narrative tenses", "question": "Select the tense used to discuss past life situations.", "type": "mcq", "options": ["Past simple", "Past perfect", "Narrative past", "Present perfect"], "answer": "Narrative past"},
    {"topic": "Future tenses", "question": "Which tense would you use to talk about an action that will be completed before a specific time in the future?", "type": "mcq", "options": ["Future perfect", "Future continuous", "Present simple", "Present continuous"], "answer": "Future perfect"},
    {"topic": "Expressing probability", "question": "Use terms like 'likely to', 'there's a good chance', 'it's doubtful'.", "type": "text"},
    {"topic": "Performance evaluation", "question": "Discuss how to agree on objectives and raise issues effectively.", "type": "text"},
    {"topic": "The Passive voice", "question": "Which sentence correctly uses passive voice?", "type": "mcq", "options": ["The book was read by her.", "She reads a book.", "She was reading a book.", "She has read the book."], "answer": "The book was read by her."},
    {"topic": "The Present Perfect", "question": "Choose the example that uses Present Perfect correctly.", "type": "mcq", "options": ["I have visited Paris last year.", "I have just visited Paris.", "I am visiting Paris.", "I visit Paris."], "answer": "I have just visited Paris."},
    {"topic": "The Future Continuous", "question": "Which example illustrates Future Continuous tense?", "type": "mcq", "options": ["I will be playing soccer.", "I played soccer.", "I play soccer.", "I am playing soccer."], "answer": "I will be playing soccer."}
]

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        self.correct_answers = 0
        self.attempted_questions = 0

        master.title("Business English Flashcards")

        self.topic_label = tk.Label(master, text="", font=('Helvetica', 14), height=4)
        self.topic_label.pack()

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

        self.next_button = tk.Button(master, text="Next", command=self.next_flashcard)
        self.next_button.pack(pady=20)

        self.next_flashcard()

    def next_flashcard(self):
        self.current_card = random.choice(flashcards)
        self.topic_label.config(text=f"Topic: {self.current_card['topic']}")
        self.display_text.config(text=f"Question: {self.current_card['question']}")
        self.user_input.delete(0, tk.END)
        self.options_frame.pack_forget()
        
        if self.current_card["type"] == "text":
            self.user_input.pack()
        elif self.current_card["type"] == "mcq":
            self.user_input.pack_forget()
            self.options_frame.pack()
            for widget in self.options_frame.winfo_children():
                widget.destroy()
            for option in self.current_card["options"]:
                tk.Radiobutton(self.options_frame, text=option, variable=self.var, value=option, font=('Helvetica', 12)).pack(anchor="w")

        self.feedback_label.config(text="")
        self.next_button.config(state="disabled")

    def check_answer(self):
        if self.current_card["type"] == "text":
            self.feedback_label.config(text="Your response has been recorded.", fg="green")
        elif self.current_card["type"] == "mcq":
            if self.var.get() == self.current_card["answer"]:
                self.feedback_label.config(text="Correct!", fg="green")
                self.correct_answers += 1
            else:
                self.feedback_label.config(text=f"Wrong! The correct answer is: {self.current_card['answer']}", fg="red")
        
        self.attempted_questions += 1
        self.next_button.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()
