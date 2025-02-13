import tkinter as tk
import random

# Extensive flashcards data: word as key and hint as value
flashcards = {
    "Future continuous": "A tense used for actions that will be ongoing at a future time.",
    "Future perfect": "A tense used to describe an action that will be completed before a specific time in the future.",
    "Employability": "Skill set making a person desirable for employment.",
    "Expertise": "Advanced knowledge or skills in a particular field.",
    "Predictions": "Forecasts about what may happen in the future based on current knowledge or evidence.",
    "Certainty": "Total assurance or conviction that something is the case.",
    "Possibility": "A thing that may happen or be the case.",
    "Implement": "To put into effect according to or by means of a definite plan or procedure.",
    "Achieve": "Successfully reach a desired objective, level, or result by effort, skill, or courage.",
    "Feedback": "Information about reactions to a product, a person's performance of a task, etc. which is used as a basis for improvement.",
    "Plan": "A detailed proposal for doing or achieving something.",
    "Planning": "The process of making plans for something.",
    "Promotion": "The action of raising someone to a higher position or rank.",
    "Employment strategies": "Tactics or methods used to secure and retain employment.",
    "Vocabulary enhancement": "The process of improving one's store of words.",
    "Discussion": "The action or process of talking about something in order to reach a decision or to exchange ideas.",
    "Predictive Analysis": "Techniques that use historical data to predict future activity.",
    "Professional growth": "The process of gaining skills, knowledge and experience both for personal development and career advancement.",
    "Performance evaluations": "Regular reviews of an employee’s job performance and overall contribution to a company.",
    "Career advancement": "The process of climbing up the career ladder; furthering one’s career.",
    "Project management": "The application of processes, methods, skills, knowledge and experience to achieve specific project objectives.",
    "Certainty factors": "Measures used to express the degree of certainty about the occurrence in uncertain situations.",
    "Development plans": "Detailed plans outlined to achieve one's long-term career objectives.",
    "Skill acquisition": "The phase during which a learner picks up a new ability.",
    "Time management": "The ability to use one's time effectively or productively, especially at work.",
    "Performance indicators": "Measurable values that demonstrate how effectively a company is achieving key business objectives.",
    "Professional effectiveness": "The degree to which one produces a decided, decisive, or desired effect in their professional life.",
    "Work-life balance": "The balance that an individual needs between time allocated for work and other aspects of life.",
    "Job satisfaction": "The feeling of fulfillment or enjoyment that comes from work.",
    "Stress management": "Ways to manage stress in personal or professional life.",
    "Leadership qualities": "The ability to lead effectively.",
    "Communication skills": "The ability to convey or share ideas and feelings effectively.",
    "Strategic thinking": "The ability to think on a broad scale, involving long-term decisions.",
    "Operational efficiency": "The capability to deliver products or services cost-effectively while ensuring quality and speed.",
    "Career goals": "Long-term aims regarding one's career.",
    "Industry insights": "Deep understandings of the trends and conditions of a specific industry.",
    "Market trends": "The general movement in the introduction, growth, maturity, and decline of products in the market.",
    "Networking": "Interacting with others to exchange information and develop professional or social contacts.",
    "Mentorship": "The guidance provided by a mentor, especially an experienced person in a company or educational institution.",
    "Innovation": "The action or process of innovating; a new method, idea, product, etc.",
    "Continuous learning": "Ongoing efforts to acquire new or existing knowledge, behaviors, skills, values or preferences."
}

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("Extensive Professional Development Vocabulary Flashcards")

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
        self.label.config(text=f"Hint for: {self.current_word}?")
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
