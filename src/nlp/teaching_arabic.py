import json
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tkinter import Tk, Label, Entry, Button, Frame, Text, Scrollbar, messagebox, StringVar, BOTH, RIGHT, Y

# Step 1: Load JSON Knowledge Base
def load_knowledge_base(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        messagebox.showerror("Error", f"File '{file_path}' not found.")
        exit(1)

# Load the knowledge base
knowledge_base_file = "arabic_grammar_data.json"
knowledge_base = load_knowledge_base(knowledge_base_file)

# Step 2: Prepare Dataset for Training
def prepare_dataset(knowledge_base):
    questions = []
    answers = []

    for letter, details in knowledge_base["alphabet"].items():
        questions.append(f"What is the Arabic letter {letter}?")
        answers.append(f"{details['name']} - Sound: {details['sound']} - Example: {details['example']}")
    for vowel in knowledge_base["vowels"]:
        questions.append(f"What is the meaning of the vowel symbol {vowel['symbol']}?")
        answers.append(f"{vowel['name']} - Pronunciation: {vowel['sound']}")
    for word in knowledge_base["words"]:
        questions.append(f"What does the word {word['word']} mean?")
        answers.append(f"{word['meaning']}")
    for rule_name, description in knowledge_base["grammar_rules"].items():
        questions.append(f"What is the meaning of {rule_name}?")
        answers.append(description)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions).toarray()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(answers)

    return X, y, vectorizer, label_encoder, questions, answers

# Prepare the dataset
X, y, vectorizer, label_encoder, original_questions, original_answers = prepare_dataset(knowledge_base)

# Step 3: Define and Train TensorFlow Model
def create_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model():
    global model  # Make model accessible to GUI functions
    model = create_model(X.shape[1], len(set(original_answers)))

    # Train the model
    model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)

    # Display training complete message
    messagebox.showinfo("Training Complete", "The model has been successfully trained!")

try:
    model = create_model(X.shape[1], len(set(original_answers)))
    model.fit(X, y, epochs=5, batch_size=8, validation_split=0.2)
except Exception:
    model = None


# Step 4: Query Using GUI
def query_knowledge_base(query):
    """
    Process the user query through the trained model and return its response.
    """
    try:
        query_vector = vectorizer.transform([query]).toarray()
        prediction = model.predict(query_vector)
        predicted_class = np.argmax(prediction)
        return label_encoder.inverse_transform([predicted_class])[0]
    except Exception as e:
        return f"Error: Cannot process query - {str(e)}"
