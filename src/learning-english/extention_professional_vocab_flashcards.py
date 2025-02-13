import tkinter as tk
import random

# Flashcards data: word as key and hint as value
flashcards = {
    "Future continuous": "Used to discuss prolonged future actions or events.",
    "Future perfect": "Used to speak about an action that will be completed before a certain future time.",
    "Employability": "Qualities and skills that make someone a desirable employee.",
    "Expertise": "Proficiency, skill, or knowledge in a particular area.",
    "Predictions": "Making claims or forecasts about future events based on data or experience.",
    "Certainty": "The state of being free from doubt.",
    "Possibility": "A thing that may happen or be the case in the future.",
    "Implement": "To start using a plan or system.",
    "Achieve": "Successfully reach a desired objective or result through effort.",
    "Feedback": "Information given back to a person about their performance of a task which is used as a basis for improvement.",
    "Planning": "The process of making plans for a future event.",
    "Promotion": "Advancement in rank or position.",
    "Employment strategies": "Approaches designed to secure employment in both the short and long term.",
    "Vocabulary enhancement": "The actions taken to improve or augment one's vocabulary.",
    # Additional terms for extended use
    "Leadership": "The action of leading a group of people or organization.",
    "Innovation": "The process of making changes to something established by introducing something new.",
    "Networking": "Interacting with others to exchange information and develop professional or social contacts.",
    "Resilience": "The capacity to recover quickly from difficulties.",
    "Negotiation": "Discussion aimed at reaching an agreement.",
    "Delegation": "The assignment of responsibility or authority to another person to carry out specific activities.",
    "Morale": "The confidence, enthusiasm, and discipline of a person or group at a particular time.",
    "Outsourcing": "Obtain (goods or a service) by contract from an outside supplier.",
    "Benchmarking": "Measure (a quality or feature) against a standard.",
    "Synergy": "The interaction of two or more organizations to produce a combined effect greater than the sum of their separate effects.",
    "Agility": "Ability to move quickly and easily",
    "Compliance": "Conforming to a rule, such as a specification, policy, standard or law.",
    "Transparency": "Operating in such a way that it is easy for others to see what actions are performed.",
    "Empowerment": "The process of becoming stronger and more confident, especially in controlling one's life and claiming one's rights.",
    "Sustainability": "The ability to be maintained at a certain rate or level.",
    "Retention": "The continued possession, use, or control of something.",
    "Acquisition": "An asset or object bought or obtained.",
    "Productivity": "The effectiveness of productive effort.",
    "Analytics": "The systematic computational analysis of data or statistics.",
    "Diversity": "A range of different things.",
    "Equity": "The quality of being fair and impartial.",
    "Onboarding": "The action or process of integrating a new employee into an organization.",
    "Mentorship": "The guidance provided by a mentor.",
    "Forecasting": "Estimate or predict (a future event or trend).",
    "Scalability": "The capacity to be changed in size or scale.",
    "Risk management": "The forecasting and evaluation of financial risks together with the identification of procedures to avoid or minimize their impact."
}

class FlashcardApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Professional Development Vocabulary Flashcards")

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
        self.label.config(text=f"Vocabulary Word: {self.current_word}\n\nEnter the definition:")
        self.hint_entry.delete(0, tk.END)
        self.result_label.config(text="")

    def check_hint(self):
        user_input = self.hint_entry.get()
        if user_input.lower().strip() == self.current_hint.lower().strip():
            self.result_label.config(text="Correct!", fg="green")
        else:
            self.result_label.config(text=f"Incorrect. The correct definition is: {self.current_hint}", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlashcardApp(root)
    root.mainloop()
