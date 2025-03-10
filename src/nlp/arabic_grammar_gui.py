import json
from tkinter import Tk, Label, Button, Entry, Text, Scrollbar, Frame, messagebox
from joblib import load
import tensorflow as tf
import numpy as np

# File paths for assets
PROCESSED_KNOWLEDGE_BASE = "processed_knowledge_base.json"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
MODEL_DIR = "saved_tensorflow_model"

# Step 1: Load the Knowledge Base and Assets
def load_saved_assets():
    """
    Load the saved knowledge base, vectorizer, label encoder, and TensorFlow model.
    """
    try:
        with open(PROCESSED_KNOWLEDGE_BASE, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", "Knowledge base not found. Ensure processed_knowledge_base.json is available.")
        exit(1)

    try:
        vectorizer = load(VECTORIZER_PATH)
    except FileNotFoundError:
        messagebox.showerror("Error", "Vectorizer not found. Ensure vectorizer.joblib is available.")
        exit(1)

    try:
        label_encoder = load(LABEL_ENCODER_PATH)
    except FileNotFoundError:
        messagebox.showerror("Error", "Label encoder not found. Ensure label_encoder.joblib is available.")
        exit(1)

    try:
        model = tf.keras.models.load_model(MODEL_DIR)
    except Exception as e:
        messagebox.showerror("Error", f"TensorFlow model not found or corrupted: {e}")
        exit(1)

    return knowledge_base, vectorizer, label_encoder, model

# Load assets
knowledge_base, vectorizer, label_encoder, model = load_saved_assets()

# Step 2: Query System
def query_system(query):
    """
    Handle user-query processing through the trained vectorizer and TensorFlow model.
    """
    try:
        query_vector = vectorizer.transform([query]).toarray()
        prediction = model.predict(query_vector)
        predicted_class = np.argmax(prediction)
        response = label_encoder.inverse_transform([predicted_class])[0]
        return response
    except Exception as e:
        return f"Error occurred while processing the query: {e}"

# Step 3: Build the GUI
def create_gui():
    """
    Build the interactive learning interface for Arabic grammar using Tkinter.
    """
    # Initialize the Tkinter window
    root = Tk()
    root.title("Arabic Grammar Learning System")
    root.geometry("600x400")

    # Header
    header_label = Label(root, text="Welcome to the Arabic Grammar Learning System!", font=("Helvetica", 16, "bold"))
    header_label.pack(pady=10)

    # Frame for query input
    input_frame = Frame(root)
    input_frame.pack(pady=10)
    
    query_label = Label(input_frame, text="Enter Your Question:")
    query_label.pack(side="left", padx=5)
    
    query_entry = Entry(input_frame, width=50)
    query_entry.pack(side="left", padx=5)

    # Output box
    output_label = Label(root, text="Response:", font=("Helvetica", 12))
    output_label.pack(pady=10)
    
    output_text = Text(root, height=10, width=70, wrap="word")
    output_text.pack(pady=10)

    # Function to process user query
    def process_query():
        user_query = query_entry.get()
        if not user_query.strip():
            messagebox.showwarning("Input Error", "Please enter a valid question!")
            return

        response = query_system(user_query.strip())
        output_text.delete(1.0, "end")  # Clear previous responses
        output_text.insert("end", response)

    # Button to submit query
    submit_button = Button(root, text="Submit", command=process_query, font=("Helvetica", 12), bg="green", fg="white")
    submit_button.pack(pady=10)

    # Footer
    footer_label = Label(root, text="Developed by Arabic Grammar AI", font=("Helvetica", 10, "italic"))
    footer_label.pack(pady=5)

    # Run the Tkinter GUI
    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
