import tkinter as tk
from tkinter import messagebox

def evaluate_expression():
    try:
        expression = entry.get()
        result = eval(expression)
        messagebox.showinfo("Result", f"The result of '{expression}' is: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid expression: {e}")

# Create the main window
root = tk.Tk()
root.title("Equation Solver")

# Create and place the entry widget
entry = tk.Entry(root, width=40)
entry.pack(padx=10, pady=10)

# Create and place the button widget
button = tk.Button(root, text="Solve", command=evaluate_expression)
button.pack(pady=5)

# Run the application
root.mainloop()
