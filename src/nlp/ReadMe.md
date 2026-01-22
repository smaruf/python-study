# NLP for Learning Arabic 🌙

A comprehensive, learner-friendly platform for learning Arabic language with focus on **communication** and **reading comprehension**. Designed for both web and mobile applications with a modern RESTful API.

## 🎯 Purpose

This platform helps learners:
- **Communicate** in Arabic through practical conversation scenarios
- **Read** and comprehend Arabic text with graded passages
- **Master** grammar rules through interactive learning
- **Practice** pronunciation and vocabulary building
- **Progress** from beginner to advanced levels

## ✨ Key Features

### 🌐 RESTful API (NEW!)
- **FastAPI-based** modern web service
- **Mobile-ready** JSON responses
- **CORS-enabled** for web applications
- **Interactive documentation** at `/docs`
- **Real-time** question answering with ML

### 📚 Learning Modules
- **28 Arabic Letters**: With pronunciation and examples
- **500+ Vocabulary Words**: Nouns, verbs, adjectives
- **Conversation Scenarios**: Greetings, shopping, directions, mosque
- **Reading Passages**: Beginner to advanced with comprehension questions
- **Grammar Rules**: Comprehensive explanations
- **Tajweed Rules**: For Quran recitation

### 🖥️ Applications
- **Web/Mobile API**: For building apps (recommended)
- **Desktop GUI**: Standalone learning application
- **Pronunciation Practice**: Speech recognition-based

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn tensorflow scikit-learn joblib numpy
```

### 2. Train the Model (First Time)
```bash
python dynamic_arabic_grammar.py
```

### 3. Start the API Server
```bash
python arabic_learning_api.py
# or
uvicorn arabic_learning_api:app --reload --port 8000
```

### 4. Access API Documentation
Open your browser: **http://localhost:8000/docs**

## 📖 API Usage Examples

### Get Vocabulary
```bash
curl "http://localhost:8000/api/vocabulary?category=verbs&limit=10"
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does السلام عليكم mean?"}'
```

### Get Conversation Scenarios
```bash
curl "http://localhost:8000/api/conversations?difficulty=beginner"
```

### Get Reading Passages
```bash
curl "http://localhost:8000/api/reading/passages?difficulty=intermediate"
```

## 📁 Project Files

### Core Application Files
- `arabic_learning_api.py` ⭐ - **RESTful API server for web/mobile (NEW)**
- `dynamic_arabic_grammar.py` - Model training script
- `arabic_grammar_gui.py` - Desktop GUI application
- `arabic_markaz_rules.py` - Pronunciation practice with speech recognition
- `teaching_arabic.py` - Alternative teaching interface

### Data Files (JSON)
- `arabic_grammar_data.json` - Core alphabet and vocabulary
- `arabic_grammar_nouns.json` - Nouns (singular/plural)
- `arabic_grammar_verbs.json` - Common verbs with examples
- `arabic_grammar_adjectives.json` - Descriptive words
- `arabic_grammar_pronouns.json` - Personal, possessive, demonstrative
- `arabic_grammar_numbers.json` - Numbers and counting
- `arabic_grammar_questions.json` - Question words
- `arabic_grammar_sentence_creation.json` - Sentence patterns
- `arabic_grammar_definite_indefinite.json` - Articles
- `arabic_grammar_sun_moon_letters.json` - Letter categories

### Documentation Files
- `LEARNING_GUIDE.md` ⭐ - **Complete platform guide (NEW)**
- `arabic_tawjeed_rules.md` - Tajweed (Quranic recitation) rules
- `arabic_markaz_rules.md` - Letter forms and positions
- `process.md` - Processing steps documentation
- `nlp_tensorflow_steps.md` - TensorFlow model details

### Generated Files (Created After Training)
- `saved_tensorflow_model/` - Trained TensorFlow model
- `vectorizer.joblib` - TF-IDF vectorizer
- `label_encoder.joblib` - Label encoder
- `processed_knowledge_base.json` - Combined knowledge base

## 🎓 Learning Path

### Beginner
1. Arabic alphabet (28 letters)
2. Basic greetings and phrases
3. Numbers 1-20
4. Simple vocabulary (family, food, colors)
5. Basic sentence structure

### Intermediate
1. Expanded vocabulary (100+ words)
2. Verb conjugations
3. Conversation practice
4. Grammar rules
5. Reading simple passages

### Advanced
1. Complex sentence construction
2. Advanced conversations
3. Tajweed rules
4. Reading comprehension
5. Cultural context

## 📱 Mobile & Web Integration

The API is designed for easy integration with mobile and web applications:

**React Native / Flutter / iOS / Android** - See examples in [LEARNING_GUIDE.md](LEARNING_GUIDE.md)

Example JavaScript fetch:
```javascript
const response = await fetch('http://your-server:8000/api/vocabulary?category=nouns');
const data = await response.json();
console.log(data.items); // Array of vocabulary items
```

## 🛠️ Technology Stack

- **FastAPI**: Modern Python web framework
- **TensorFlow**: Machine learning for Q&A
- **scikit-learn**: Text vectorization and classification
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation
- **JSON**: Data storage format

## 📚 Complete Documentation

For comprehensive documentation including:
- All API endpoints and examples
- Mobile integration guides
- Data structure details
- Contributing guidelines
- Security best practices

**See [LEARNING_GUIDE.md](LEARNING_GUIDE.md)**

## 🌟 What's New

### Version 1.0 (January 2026)
- ✅ RESTful API with FastAPI
- ✅ Conversation practice scenarios
- ✅ Reading comprehension passages
- ✅ Mobile-friendly JSON responses
- ✅ Interactive API documentation
- ✅ CORS support for web apps
- ✅ Comprehensive learning guide
- ✅ Difficulty levels (beginner/intermediate/advanced)

## 🤝 Contributing

We welcome contributions! Ways to help:
- Add more vocabulary and phrases
- Improve translations
- Add new conversation scenarios
- Create reading passages
- Report bugs or suggest features

## 📞 Support

- Check [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for detailed documentation
- API docs at `http://localhost:8000/docs` when server is running
- Open an issue for bugs or feature requests

## 📄 License

This project is licensed under the MIT License.

---

**🌙 Made for Arabic learners worldwide - focusing on practical communication and reading skills**

*For web and mobile applications - Start learning Arabic today!*
