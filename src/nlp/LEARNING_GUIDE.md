# Arabic Learning Platform - Complete Guide

## 🌟 Overview

This is a comprehensive Arabic learning platform designed for learners who want to improve their communication skills and reading comprehension in Arabic. The platform provides both a desktop GUI application and a RESTful API for web and mobile integration.

## 📋 Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Documentation](#api-documentation)
5. [Desktop Application](#desktop-application)
6. [Learning Modules](#learning-modules)
7. [Data Structure](#data-structure)
8. [For Developers](#for-developers)
9. [Mobile Integration](#mobile-integration)
10. [Contributing](#contributing)

## ✨ Features

### For Learners
- **Interactive Vocabulary Learning**: Learn Arabic letters, words, and phrases with examples
- **Grammar Rules**: Comprehensive grammar explanations with practical examples
- **Conversation Practice**: Real-world conversation scenarios for daily communication
- **Reading Comprehension**: Graded reading passages with translations and questions
- **Pronunciation Guide**: Transliteration and pronunciation tips
- **Progress Tracking**: Track your learning journey

### For Developers
- **RESTful API**: FastAPI-based API for web and mobile integration
- **Machine Learning**: TensorFlow-powered question answering system
- **JSON Data Format**: Easy to extend and customize
- **CORS Support**: Ready for cross-origin requests
- **Interactive Documentation**: Auto-generated API docs with Swagger UI

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
# Basic dependencies for the API
pip install fastapi uvicorn pydantic

# Machine Learning dependencies
pip install tensorflow scikit-learn joblib numpy

# Desktop GUI dependencies (optional)
pip install tk

# Pronunciation practice dependencies (optional)
pip install speechrecognition transformers torch
```

Or install all at once:

```bash
pip install fastapi uvicorn pydantic tensorflow scikit-learn joblib numpy tk speechrecognition transformers torch
```

### Step 2: Train the Model (First Time Only)

Before using the API or GUI, you need to train the machine learning model:

```bash
cd /path/to/python-study/src/nlp
python dynamic_arabic_grammar.py
```

This will:
- Load all Arabic grammar JSON files
- Create a combined knowledge base
- Train a TensorFlow model
- Save the model and vectorizers for future use

## 🎯 Quick Start

### Option 1: RESTful API Server (Recommended for Web/Mobile)

```bash
# Start the API server
cd /path/to/python-study/src/nlp
python arabic_learning_api.py

# Or use uvicorn directly
uvicorn arabic_learning_api:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive API documentation at: **http://localhost:8000/docs**

### Option 2: Desktop GUI Application

```bash
cd /path/to/python-study/src/nlp
python arabic_grammar_gui.py
```

### Option 3: Pronunciation Practice

```bash
cd /path/to/python-study/src/nlp
python arabic_markaz_rules.py
```

## 📚 API Documentation

### Base URL

```
http://localhost:8000
```

### Core Endpoints

#### 1. Query Knowledge Base

Ask any question about Arabic grammar or vocabulary:

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Arabic letter ب?"}'
```

Response:
```json
{
  "question": "What is the Arabic letter ب?",
  "answer": "Ba - Sound: B - Example: بيت (Bayt - House)",
  "confidence": 0.95
}
```

#### 2. Get Vocabulary

Retrieve vocabulary by category:

```bash
# Get nouns
curl "http://localhost:8000/api/vocabulary?category=nouns&limit=10"

# Get verbs
curl "http://localhost:8000/api/vocabulary?category=verbs&limit=10"

# Get adjectives
curl "http://localhost:8000/api/vocabulary?category=adjectives&limit=10"
```

Response:
```json
{
  "category": "nouns",
  "count": 10,
  "items": [
    {"word": "كتاب", "meaning": "Book"},
    {"word": "بنت", "meaning": "Girl"},
    {"word": "ولد", "meaning": "Boy"}
  ]
}
```

#### 3. Arabic Alphabet

Get the complete Arabic alphabet:

```bash
curl "http://localhost:8000/api/alphabet"
```

#### 4. Conversation Scenarios

Get practical conversation phrases:

```bash
# All scenarios
curl "http://localhost:8000/api/conversations"

# Beginner level only
curl "http://localhost:8000/api/conversations?difficulty=beginner"
```

Response:
```json
{
  "count": 5,
  "scenarios": [
    {
      "scenario": "Greetings and Introductions",
      "difficulty": "beginner",
      "phrases": [
        {
          "arabic": "السلام عليكم",
          "english": "Peace be upon you (Hello)",
          "transliteration": "As-salamu alaykum"
        }
      ]
    }
  ]
}
```

#### 5. Reading Passages

Get reading comprehension exercises:

```bash
# All passages
curl "http://localhost:8000/api/reading/passages"

# Beginner passages only
curl "http://localhost:8000/api/reading/passages?difficulty=beginner&limit=3"
```

Response:
```json
{
  "count": 3,
  "passages": [
    {
      "id": "passage_1",
      "title": "My Family (عائلتي)",
      "difficulty": "beginner",
      "text_arabic": "أنا محمد. عائلتي كبيرة...",
      "text_english": "I am Mohammed. My family is big...",
      "vocabulary": [...],
      "questions": [...]
    }
  ]
}
```

#### 6. Grammar Rules

Learn Arabic grammar:

```bash
# All grammar rules
curl "http://localhost:8000/api/grammar/rules"

# Specific rule
curl "http://localhost:8000/api/grammar/rules?rule=noun_definition"
```

#### 7. Numbers

Learn Arabic numbers:

```bash
curl "http://localhost:8000/api/numbers"
```

#### 8. Pronouns

Get Arabic pronouns:

```bash
curl "http://localhost:8000/api/pronouns"
```

### Complete API Reference

Visit **http://localhost:8000/docs** for complete interactive API documentation with:
- All available endpoints
- Request/response schemas
- Try-it-out functionality
- Example requests and responses

## 🖥️ Desktop Application

The GUI application provides an interactive learning interface:

### Features:
- **Question-Answer System**: Ask questions about Arabic grammar
- **Visual Interface**: User-friendly Tkinter-based GUI
- **Real-time Responses**: Instant answers using the trained ML model

### Usage:

1. Run the application:
```bash
python arabic_grammar_gui.py
```

2. Enter your question in the text field
3. Click "Submit" to get an answer
4. View the response in the output area

## 📖 Learning Modules

### 1. Alphabet & Letters
- 28 Arabic letters
- Pronunciation guide
- Word examples for each letter
- Letter forms (isolated, initial, medial, final)

### 2. Vocabulary
- **Nouns**: Common objects, places, and concepts
- **Verbs**: Actions and states
- **Adjectives**: Descriptive words
- **Numbers**: Cardinal and ordinal numbers
- **Pronouns**: Personal, possessive, demonstrative

### 3. Grammar
- Definite and indefinite articles
- Masculine and feminine forms
- Singular and plural forms
- Sentence construction patterns
- Sun and moon letters

### 4. Conversation Practice
- Greetings and introductions
- Shopping and dining
- Asking for directions
- Daily conversations
- Mosque-related phrases

### 5. Reading Comprehension
- Beginner: Simple sentences about family, daily life
- Intermediate: Paragraphs about routines, activities
- Advanced: Complex texts with detailed descriptions

### 6. Tajweed (Quranic Recitation)
- Makhraj (articulation points)
- Sifaat (letter characteristics)
- Rules of Noon Saakinah and Tanween
- Qalqalah (echoing)
- Complete guide in `arabic_tawjeed_rules.md`

## 📁 Data Structure

### JSON Files

The platform uses multiple JSON files for organized data:

- `arabic_grammar_data.json`: Core alphabet and basic vocabulary
- `arabic_grammar_nouns.json`: Nouns (singular/plural)
- `arabic_grammar_verbs.json`: Common verbs with examples
- `arabic_grammar_adjectives.json`: Descriptive words
- `arabic_grammar_pronouns.json`: All pronoun types
- `arabic_grammar_numbers.json`: Numbers and counting
- `arabic_grammar_questions.json`: Question words and phrases
- `arabic_grammar_sentence_creation.json`: Sentence patterns
- `arabic_grammar_definite_indefinite.json`: Articles
- `arabic_grammar_sun_moon_letters.json`: Letter categories

### Adding Custom Data

To add new vocabulary or grammar rules:

1. Open the relevant JSON file
2. Follow the existing structure
3. Add your new entries
4. Retrain the model:
```bash
python dynamic_arabic_grammar.py
```

Example - Adding a new verb:

```json
{
  "verbs": [
    {
      "word": "درس",
      "meaning": "Studied",
      "example": "الطالب درس بجد."
    }
  ]
}
```

## 👨‍💻 For Developers

### Project Structure

```
nlp/
├── arabic_learning_api.py          # FastAPI server (NEW)
├── dynamic_arabic_grammar.py       # Model training script
├── arabic_grammar_gui.py           # Desktop GUI application
├── teaching_arabic.py              # Alternative teaching interface
├── arabic_markaz_rules.py          # Pronunciation practice
├── arabic_grammar_*.json           # Data files
├── arabic_tawjeed_rules.md         # Tajweed documentation
├── arabic_markaz_rules.md          # Letter forms documentation
└── ReadMe.md                       # This file
```

### API Architecture

```
┌─────────────────┐
│  Mobile/Web App │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   FastAPI Server│
│  (Port 8000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  TensorFlow Model + Vectorizer  │
│  + JSON Knowledge Base          │
└─────────────────────────────────┘
```

### Technology Stack

- **Backend**: FastAPI, Uvicorn
- **ML/AI**: TensorFlow, scikit-learn
- **Data**: JSON, joblib (serialization)
- **GUI**: Tkinter
- **Speech**: SpeechRecognition, Wav2Vec2 (optional)

### Adding New Endpoints

Example - Add a new endpoint:

```python
@app.get("/api/phrases/daily", tags=["Phrases"])
async def get_daily_phrases():
    """Get commonly used daily phrases"""
    return {
        "phrases": [
            {"arabic": "صباح الخير", "english": "Good morning"},
            {"arabic": "مساء الخير", "english": "Good evening"}
        ]
    }
```

### Testing the API

Use pytest or manual testing:

```bash
# Health check
curl http://localhost:8000/api/health

# Test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does السلام عليكم mean?"}'
```

## 📱 Mobile Integration

### React Native Example

```javascript
// Fetch vocabulary
const getVocabulary = async () => {
  const response = await fetch(
    'http://your-server:8000/api/vocabulary?category=nouns&limit=20'
  );
  const data = await response.json();
  return data.items;
};

// Ask a question
const askQuestion = async (question) => {
  const response = await fetch(
    'http://your-server:8000/api/query',
    {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question})
    }
  );
  const data = await response.json();
  return data.answer;
};
```

### Flutter Example

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

// Get conversation scenarios
Future<List> getConversations() async {
  final response = await http.get(
    Uri.parse('http://your-server:8000/api/conversations?difficulty=beginner')
  );
  
  if (response.statusCode == 200) {
    final data = json.decode(response.body);
    return data['scenarios'];
  }
  throw Exception('Failed to load conversations');
}
```

### iOS (Swift) Example

```swift
// Fetch reading passages
func getReadingPassages(difficulty: String) {
    let url = URL(string: "http://your-server:8000/api/reading/passages?difficulty=\(difficulty)")!
    
    URLSession.shared.dataTask(with: url) { data, response, error in
        if let data = data {
            let passages = try? JSONDecoder().decode(PassagesResponse.self, from: data)
            // Use passages
        }
    }.resume()
}
```

## 🎓 Learning Path

### Beginner (Weeks 1-4)
1. Learn the Arabic alphabet
2. Practice basic greetings
3. Learn numbers 1-20
4. Study common nouns (20-30 words)
5. Read simple passages about family

### Intermediate (Weeks 5-12)
1. Expand vocabulary (100+ words)
2. Learn verb conjugations
3. Practice conversation scenarios
4. Study grammar rules
5. Read intermediate passages

### Advanced (Weeks 13+)
1. Master sentence construction
2. Practice advanced conversations
3. Study Tajweed rules
4. Read complex texts
5. Focus on reading comprehension

## 🔒 Security Notes

For production deployment:

1. **API Authentication**: Add JWT or API key authentication
2. **CORS Configuration**: Restrict allowed origins
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **HTTPS**: Use SSL/TLS for encrypted communication
5. **Input Validation**: Validate all user inputs

Example - Adding authentication:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Add your token verification logic
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials
```

## 🤝 Contributing

Contributions are welcome! You can help by:

1. **Adding More Data**: Contribute new vocabulary, phrases, or passages
2. **Improving Translations**: Ensure accuracy of Arabic translations
3. **Adding Features**: Implement new API endpoints or UI features
4. **Bug Fixes**: Report and fix bugs
5. **Documentation**: Improve this guide and code comments

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review API documentation at `/docs`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Arabic language data compiled from various educational sources
- TensorFlow and scikit-learn for ML capabilities
- FastAPI framework for modern API development
- The open-source community

---

**Made with ❤️ for Arabic learners worldwide**

*Last Updated: January 2026*
