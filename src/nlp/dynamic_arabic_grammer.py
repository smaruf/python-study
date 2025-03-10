import json
import os
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load

# File paths
MODEL_DIR = "saved_tensorflow_model"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
PROCESSED_KNOWLEDGE_BASE = "processed_knowledge_base.json"

# Step 1: Load and merge JSON files dynamically
def load_knowledge_base(prefix="arabic_grammar_"):
    """
    Dynamically load and merge knowledge base from multiple JSON files matching the prefix.
    """
    knowledge_base = {"alphabet": {}, "nouns": {}, "pronouns": {}, "adjectives": {}, "sentence_construction": {}, "grammar_rules": {}}
    json_files = glob(f"{prefix}*.json")
    
    if not json_files:
        print(f"No JSON files found with prefix '{prefix}'")
        return None

    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Merge sections into the knowledge base
                for section, content in data.items():
                    if section in knowledge_base and isinstance(content, (list, dict)):
                        if isinstance(content, list):  # Append lists
                            if section not in knowledge_base:
                                knowledge_base[section] = []
                            knowledge_base[section].extend(content)
                        elif isinstance(content, dict):  # Update dictionaries
                            knowledge_base[section].update(content)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    return knowledge_base

# Step 2: Prepare Dataset for Model Training
def prepare_dataset(knowledge_base):
    """
    Convert the knowledge base into a question-answer dataset for training.
    """
    if not knowledge_base:
        raise ValueError("Knowledge base is empty or None.")
    
    questions = []
    answers = []

    # Alphabet
    for letter, details in knowledge_base["alphabet"].items():
        questions.append(f"What is the Arabic letter {letter}?")
        answers.append(f"{details['name']} - Sound: {details['sound']} - Example: {details['example']}")

    # Singular and plural nouns
    for category, values in knowledge_base["nouns"].items():
        for entry in values:
            questions.append(f"What does the noun {entry['word']} mean?")
            answers.append(entry['meaning'])

    # Pronouns
    for category, pronouns in knowledge_base["pronouns"].items():
        for pronoun, translation in pronouns.items():
            questions.append(f"What is the Arabic pronoun {pronoun}?")
            answers.append(translation)

    # Adjectives
    for adjective in knowledge_base["adjectives"]["basic"]:
        questions.append(f"What does the adjective {adjective['word']} mean?")
        answers.append(adjective['meaning'])

    # Sentence Construction
    for sentence in knowledge_base["sentence_construction"]["examples"]:
        questions.append(f"What is the translation of '{sentence['sentence']}'?")
        answers.append(sentence["translation"])

    # Grammar Rules
    for rule, description in knowledge_base["grammar_rules"].items():
        questions.append(f"Explain the grammar rule '{rule}'.")
        answers.append(description)

    return questions, answers

# Step 3: Build or Load TF-IDF Vectorizer and Label Encoder
def build_vectorizer_and_encoder(questions, answers):
    """
    Build or load the TF-IDF vectorizer and label encoder, and fit them to the dataset.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions).toarray()
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(answers)
    
    return X, y, vectorizer, label_encoder

# Step 4: Build and Train TensorFlow Model
def create_model(input_dim, output_dim):
    """
    Create a simple feedforward neural network using TensorFlow.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_or_load_model(X, y):
    """
    Train a new TensorFlow model or load an existing one.
    """
    if os.path.exists(MODEL_DIR):
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_DIR)
    else:
        print("Training a new model...")
        model = create_model(X.shape[1], len(set(y)))
        model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)
        model.save(MODEL_DIR)
    return model

# Step 5: Save Processed Assets
def save_assets(vectorizer, label_encoder, questions, answers):
    """
    Save vectorizer, label encoder, and processed knowledge base.
    """
    dump(vectorizer, VECTORIZER_PATH)
    dump(label_encoder, LABEL_ENCODER_PATH)
    with open(PROCESSED_KNOWLEDGE_BASE, "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers}, f, ensure_ascii=False, indent=4)

# Step 6: Query the Knowledge Base
def query_system(query, model, vectorizer, label_encoder):
    """
    Process a user query through the vectorizer and TensorFlow model to generate a response.
    """
    query_vector = vectorizer.transform([query]).toarray()
    prediction = model.predict(query_vector)
    predicted_class = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_class])[0]

# Main Program
def main():
    # Load and merge all JSON files
    knowledge_base = load_knowledge_base()

    if not knowledge_base:
        print("No knowledge base available. Exiting.")
        return

    # Prepare dataset
    questions, answers = prepare_dataset(knowledge_base)

    # Build TF-IDF vectorizer and label encoder
    X, y, vectorizer, label_encoder = build_vectorizer_and_encoder(questions, answers)

    # Train or load TensorFlow model
    model = train_or_load_model(X, y)

    # Save assets for future use
    save_assets(vectorizer, label_encoder, questions, answers)

    print("System is ready! Start asking questions.")

    # Interactive Question-Answer Loop
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = query_system(user_query, model, vectorizer, label_encoder)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
