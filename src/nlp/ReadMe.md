# NLP for learning Arabic

This directory contains scripts and modules for Natural Language Processing (NLP) tasks, focusing on Arabic grammar.

## Contents

- `dynamic_arabic_grammar.py`: Loads and processes Arabic grammar data from JSON files, then trains a machine learning model.
- `saved_tensorflow_model`: Directory for the trained TensorFlow model.
- `vectorizer.joblib`: Serialized TF-IDF vectorizer.
- `label_encoder.joblib`: Serialized label encoder.
- `processed_knowledge_base.json`: Combined knowledge base from multiple JSON files.

## Usage

To run the script and train the model, execute:
```
python dynamic_arabic_grammar.py
```

Ensure all required dependencies are installed. You can install them using:
```
pip install numpy tensorflow scikit-learn joblib
```

For detailed processing steps, refer to [process.md](process.md).

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
