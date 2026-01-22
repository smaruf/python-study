"""
Arabic Learning API Server
==========================

This FastAPI-based RESTful API provides endpoints for learning Arabic language
with a focus on communication and reading comprehension.

Features:
- Vocabulary and grammar queries
- Conversation practice scenarios
- Reading comprehension exercises
- Progress tracking
- Multi-level difficulty support

Requirements:
- FastAPI
- Uvicorn
- TensorFlow
- scikit-learn
- joblib

Installation:
pip install fastapi uvicorn tensorflow scikit-learn joblib

Usage:
uvicorn arabic_learning_api:app --reload --host 0.0.0.0 --port 8000

API Documentation:
Once running, visit http://localhost:8000/docs for interactive API documentation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from joblib import load

# Initialize FastAPI app
app = FastAPI(
    title="Arabic Learning API",
    description="RESTful API for learning Arabic with focus on communication and reading",
    version="1.0.0"
)

# Enable CORS for web and mobile applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
MODEL_DIR = "saved_tensorflow_model"
VECTORIZER_PATH = "vectorizer.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"
PROCESSED_KNOWLEDGE_BASE = "processed_knowledge_base.json"

# Global variables for loaded models
vectorizer = None
label_encoder = None
model = None
knowledge_base = {}


# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    """Request model for querying the knowledge base"""
    question: str = Field(..., description="Question about Arabic grammar or vocabulary")
    
class QueryResponse(BaseModel):
    """Response model for query results"""
    question: str
    answer: str
    confidence: float = Field(..., description="Confidence score between 0 and 1")

class VocabularyItem(BaseModel):
    """Model for vocabulary items"""
    word: str
    meaning: str
    example: Optional[str] = None
    
class ConversationScenario(BaseModel):
    """Model for conversation practice scenarios"""
    scenario: str
    phrases: List[Dict[str, str]]
    difficulty: str = Field(..., description="beginner, intermediate, or advanced")

class ReadingPassage(BaseModel):
    """Model for reading comprehension passages"""
    title: str
    text_arabic: str
    text_english: str
    difficulty: str
    questions: List[Dict[str, Any]]

class ProgressUpdate(BaseModel):
    """Model for tracking learner progress"""
    user_id: str
    lesson_id: str
    score: float = Field(..., ge=0, le=100)
    completed: bool


# Utility functions
def load_knowledge_base_files(prefix="arabic_grammar_"):
    """
    Dynamically load and merge knowledge base from multiple JSON files.
    
    Returns:
        dict: Merged knowledge base from all JSON files
    """
    kb = {
        "alphabet": {}, 
        "nouns": {}, 
        "pronouns": {}, 
        "adjectives": {}, 
        "verbs": [],
        "sentence_creation": {},
        "sentence_construction": {},
        "grammar_rules": {},
        "questions": {},
        "numbers": {},
        "sun_moon_letters": {},
        "definite_indefinite": {}
    }
    
    json_files = glob(f"{prefix}*.json")
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for section, content in data.items():
                    if section in kb:
                        if isinstance(content, list):
                            if not isinstance(kb[section], list):
                                kb[section] = []
                            kb[section].extend(content)
                        elif isinstance(content, dict):
                            if not isinstance(kb[section], dict):
                                kb[section] = {}
                            kb[section].update(content)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    return kb


def load_models():
    """Load trained models and vectorizers if available"""
    global vectorizer, label_encoder, model, knowledge_base
    
    # Load knowledge base
    knowledge_base = load_knowledge_base_files()
    
    # Try to load trained model components
    try:
        if os.path.exists(VECTORIZER_PATH):
            vectorizer = load(VECTORIZER_PATH)
        if os.path.exists(LABEL_ENCODER_PATH):
            label_encoder = load(LABEL_ENCODER_PATH)
        if os.path.exists(MODEL_DIR):
            model = tf.keras.models.load_model(MODEL_DIR)
    except Exception as e:
        print(f"Warning: Could not load trained models: {e}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    load_models()
    print("Arabic Learning API started successfully!")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Arabic Learning API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "vocabulary": "/api/vocabulary",
            "grammar": "/api/grammar",
            "conversations": "/api/conversations",
            "reading": "/api/reading",
            "query": "/api/query"
        }
    }


@app.get("/api/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "vectorizer": vectorizer is not None,
            "label_encoder": label_encoder is not None,
            "model": model is not None,
            "knowledge_base": len(knowledge_base) > 0
        }
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_knowledge(request: QueryRequest):
    """
    Query the Arabic learning knowledge base with natural language questions.
    
    Example questions:
    - "What is the Arabic letter ب?"
    - "What does the word كتاب mean?"
    - "How do I say 'thank you' in Arabic?"
    """
    if not model or not vectorizer or not label_encoder:
        raise HTTPException(
            status_code=503, 
            detail="AI model not available. Please ensure the model is trained."
        )
    
    try:
        query_vector = vectorizer.transform([request.question]).toarray()
        prediction = model.predict(query_vector, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        answer = label_encoder.inverse_transform([predicted_class])[0]
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/vocabulary", tags=["Vocabulary"])
async def get_vocabulary(
    category: Optional[str] = Query(None, description="Category: nouns, verbs, adjectives"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return")
):
    """
    Get vocabulary items by category.
    
    Categories:
    - nouns: Common Arabic nouns
    - verbs: Common Arabic verbs
    - adjectives: Descriptive words
    """
    result = []
    
    if category == "nouns" and "nouns" in knowledge_base:
        for cat, items in knowledge_base["nouns"].items():
            if isinstance(items, list):
                result.extend(items[:limit])
    elif category == "verbs" and "verbs" in knowledge_base:
        result = knowledge_base["verbs"][:limit]
    elif category == "adjectives" and "adjectives" in knowledge_base:
        if "basic" in knowledge_base["adjectives"]:
            result = knowledge_base["adjectives"]["basic"][:limit]
        else:
            result = knowledge_base["adjectives"][:limit] if isinstance(knowledge_base["adjectives"], list) else []
    else:
        # Return mixed vocabulary
        if "nouns" in knowledge_base and "singular" in knowledge_base["nouns"]:
            result.extend(knowledge_base["nouns"]["singular"][:10])
        if "verbs" in knowledge_base:
            result.extend(knowledge_base["verbs"][:10])
    
    return {
        "category": category or "mixed",
        "count": len(result),
        "items": result[:limit]
    }


@app.get("/api/alphabet", tags=["Vocabulary"])
async def get_alphabet():
    """Get the complete Arabic alphabet with pronunciation and examples"""
    if "alphabet" not in knowledge_base:
        raise HTTPException(status_code=404, detail="Alphabet data not found")
    
    return {
        "count": len(knowledge_base["alphabet"]),
        "letters": knowledge_base["alphabet"]
    }


@app.get("/api/grammar/rules", tags=["Grammar"])
async def get_grammar_rules(
    rule: Optional[str] = Query(None, description="Specific grammar rule name")
):
    """
    Get Arabic grammar rules and explanations.
    
    If no specific rule is provided, returns all available rules.
    """
    if "grammar_rules" not in knowledge_base:
        raise HTTPException(status_code=404, detail="Grammar rules not found")
    
    if rule:
        if rule in knowledge_base["grammar_rules"]:
            return {
                "rule": rule,
                "explanation": knowledge_base["grammar_rules"][rule]
            }
        else:
            raise HTTPException(status_code=404, detail=f"Grammar rule '{rule}' not found")
    
    return {
        "count": len(knowledge_base["grammar_rules"]),
        "rules": knowledge_base["grammar_rules"]
    }


@app.get("/api/grammar/sentence-construction", tags=["Grammar"])
async def get_sentence_construction():
    """Get examples of Arabic sentence construction patterns"""
    data = knowledge_base.get("sentence_construction") or knowledge_base.get("sentence_creation")
    
    if not data:
        raise HTTPException(status_code=404, detail="Sentence construction data not found")
    
    return {
        "sentence_patterns": data
    }


@app.get("/api/conversations", tags=["Conversation"])
async def get_conversation_scenarios(
    difficulty: Optional[str] = Query(None, description="beginner, intermediate, or advanced")
):
    """
    Get conversation practice scenarios for different situations.
    
    Includes common phrases for:
    - Greetings and introductions
    - Shopping and dining
    - Asking for directions
    - Daily conversations
    """
    scenarios = [
        {
            "scenario": "Greetings and Introductions",
            "difficulty": "beginner",
            "phrases": [
                {"arabic": "السلام عليكم", "english": "Peace be upon you (Hello)", "transliteration": "As-salamu alaykum"},
                {"arabic": "وعليكم السلام", "english": "And upon you peace (Hello response)", "transliteration": "Wa alaykum as-salam"},
                {"arabic": "كيف حالك؟", "english": "How are you?", "transliteration": "Kayfa haluk?"},
                {"arabic": "بخير، الحمد لله", "english": "Fine, praise be to God", "transliteration": "Bi-khayr, alhamdulillah"},
                {"arabic": "ما اسمك؟", "english": "What is your name?", "transliteration": "Ma ismuka?"},
                {"arabic": "اسمي...", "english": "My name is...", "transliteration": "Ismi..."},
                {"arabic": "تشرفت بمعرفتك", "english": "Nice to meet you", "transliteration": "Tasharraftu bi-ma'rifatik"}
            ]
        },
        {
            "scenario": "Shopping and Dining",
            "difficulty": "intermediate",
            "phrases": [
                {"arabic": "كم الثمن؟", "english": "How much is it?", "transliteration": "Kam ath-thaman?"},
                {"arabic": "أريد أن أشتري...", "english": "I want to buy...", "transliteration": "Uridu an ashtari..."},
                {"arabic": "هل عندك...؟", "english": "Do you have...?", "transliteration": "Hal 'indaka...?"},
                {"arabic": "أريد القائمة من فضلك", "english": "I want the menu please", "transliteration": "Uridu al-qa'imah min fadlik"},
                {"arabic": "الحساب من فضلك", "english": "The bill please", "transliteration": "Al-hisab min fadlik"},
                {"arabic": "شكراً جزيلاً", "english": "Thank you very much", "transliteration": "Shukran jazilan"}
            ]
        },
        {
            "scenario": "Asking for Directions",
            "difficulty": "intermediate",
            "phrases": [
                {"arabic": "أين...؟", "english": "Where is...?", "transliteration": "Ayna...?"},
                {"arabic": "كيف أصل إلى...؟", "english": "How do I get to...?", "transliteration": "Kayfa asil ila...?"},
                {"arabic": "على اليمين", "english": "On the right", "transliteration": "Ala al-yamin"},
                {"arabic": "على اليسار", "english": "On the left", "transliteration": "Ala al-yasar"},
                {"arabic": "مباشرة", "english": "Straight ahead", "transliteration": "Mubasharatan"},
                {"arabic": "قريب من هنا", "english": "Close from here", "transliteration": "Qarib min huna"},
                {"arabic": "بعيد من هنا", "english": "Far from here", "transliteration": "Ba'id min huna"}
            ]
        },
        {
            "scenario": "Daily Conversations",
            "difficulty": "beginner",
            "phrases": [
                {"arabic": "نعم", "english": "Yes", "transliteration": "Na'am"},
                {"arabic": "لا", "english": "No", "transliteration": "La"},
                {"arabic": "من فضلك", "english": "Please", "transliteration": "Min fadlik"},
                {"arabic": "شكراً", "english": "Thank you", "transliteration": "Shukran"},
                {"arabic": "عفواً", "english": "You're welcome", "transliteration": "Afwan"},
                {"arabic": "آسف", "english": "Sorry", "transliteration": "Asif"},
                {"arabic": "مع السلامة", "english": "Goodbye", "transliteration": "Ma'a as-salamah"}
            ]
        },
        {
            "scenario": "At the Mosque",
            "difficulty": "intermediate",
            "phrases": [
                {"arabic": "أين المسجد؟", "english": "Where is the mosque?", "transliteration": "Ayna al-masjid?"},
                {"arabic": "متى الصلاة؟", "english": "When is the prayer?", "transliteration": "Mata as-salah?"},
                {"arabic": "جزاك الله خيراً", "english": "May God reward you with good", "transliteration": "Jazaka Allahu khayran"},
                {"arabic": "بارك الله فيك", "english": "May God bless you", "transliteration": "Baraka Allahu fik"}
            ]
        }
    ]
    
    if difficulty:
        scenarios = [s for s in scenarios if s["difficulty"] == difficulty.lower()]
    
    return {
        "count": len(scenarios),
        "scenarios": scenarios
    }


@app.get("/api/reading/passages", tags=["Reading"])
async def get_reading_passages(
    difficulty: Optional[str] = Query(None, description="beginner, intermediate, or advanced"),
    limit: int = Query(5, ge=1, le=20)
):
    """
    Get reading comprehension passages with translations and questions.
    
    Includes stories and passages at different difficulty levels to improve
    reading skills and comprehension.
    """
    passages = [
        {
            "id": "passage_1",
            "title": "My Family (عائلتي)",
            "difficulty": "beginner",
            "text_arabic": "أنا محمد. عائلتي كبيرة. أبي معلم وأمي طبيبة. لي أخ واحد وأختان. أخي طالب في الجامعة. أختاي تدرسان في المدرسة.",
            "text_english": "I am Mohammed. My family is big. My father is a teacher and my mother is a doctor. I have one brother and two sisters. My brother is a student at the university. My two sisters study at school.",
            "vocabulary": [
                {"word": "عائلتي", "meaning": "my family"},
                {"word": "كبيرة", "meaning": "big"},
                {"word": "أبي", "meaning": "my father"},
                {"word": "أمي", "meaning": "my mother"},
                {"word": "أخ", "meaning": "brother"},
                {"word": "أخت", "meaning": "sister"}
            ],
            "questions": [
                {"question": "What does Mohammed's father do?", "answer": "He is a teacher"},
                {"question": "How many sisters does Mohammed have?", "answer": "Two sisters"},
                {"question": "Where does Mohammed's brother study?", "answer": "At the university"}
            ]
        },
        {
            "id": "passage_2",
            "title": "Daily Routine (روتيني اليومي)",
            "difficulty": "intermediate",
            "text_arabic": "أستيقظ كل يوم في الساعة السابعة صباحاً. أصلي الفجر ثم أتناول الإفطار مع عائلتي. أذهب إلى العمل في الساعة الثامنة. أعمل في مكتب من التاسعة حتى الخامسة مساءً. بعد العمل، أعود إلى البيت وأقضي الوقت مع عائلتي.",
            "text_english": "I wake up every day at seven in the morning. I pray Fajr and then have breakfast with my family. I go to work at eight o'clock. I work in an office from nine until five in the evening. After work, I return home and spend time with my family.",
            "vocabulary": [
                {"word": "أستيقظ", "meaning": "I wake up"},
                {"word": "أصلي", "meaning": "I pray"},
                {"word": "أتناول", "meaning": "I have/eat"},
                {"word": "أذهب", "meaning": "I go"},
                {"word": "أعمل", "meaning": "I work"},
                {"word": "أعود", "meaning": "I return"}
            ],
            "questions": [
                {"question": "What time does the person wake up?", "answer": "Seven in the morning"},
                {"question": "What does the person do before breakfast?", "answer": "Prays Fajr"},
                {"question": "How long does the person work?", "answer": "From nine until five"}
            ]
        },
        {
            "id": "passage_3",
            "title": "The Market (السوق)",
            "difficulty": "beginner",
            "text_arabic": "ذهبت إلى السوق اليوم. اشتريت خبزاً وفاكهة وخضروات. السوق كان مزدحماً جداً. التفاح طازج والسعر جيد. دفعت ثلاثين درهماً.",
            "text_english": "I went to the market today. I bought bread, fruit, and vegetables. The market was very crowded. The apples were fresh and the price was good. I paid thirty dirhams.",
            "vocabulary": [
                {"word": "السوق", "meaning": "the market"},
                {"word": "اشتريت", "meaning": "I bought"},
                {"word": "خبز", "meaning": "bread"},
                {"word": "فاكهة", "meaning": "fruit"},
                {"word": "مزدحم", "meaning": "crowded"},
                {"word": "طازج", "meaning": "fresh"}
            ],
            "questions": [
                {"question": "What did the person buy?", "answer": "Bread, fruit, and vegetables"},
                {"question": "How was the market?", "answer": "Very crowded"},
                {"question": "How much did the person pay?", "answer": "Thirty dirhams"}
            ]
        }
    ]
    
    if difficulty:
        passages = [p for p in passages if p["difficulty"] == difficulty.lower()]
    
    return {
        "count": len(passages[:limit]),
        "passages": passages[:limit]
    }


@app.get("/api/numbers", tags=["Vocabulary"])
async def get_numbers():
    """Get Arabic numbers with pronunciation"""
    if "numbers" in knowledge_base and knowledge_base["numbers"]:
        return {
            "numbers": knowledge_base["numbers"]
        }
    
    # Default number data if not in knowledge base
    numbers = {
        "cardinal": [
            {"number": 0, "arabic": "صفر", "transliteration": "sifr"},
            {"number": 1, "arabic": "واحد", "transliteration": "wahid"},
            {"number": 2, "arabic": "اثنان", "transliteration": "ithnan"},
            {"number": 3, "arabic": "ثلاثة", "transliteration": "thalatha"},
            {"number": 4, "arabic": "أربعة", "transliteration": "arba'a"},
            {"number": 5, "arabic": "خمسة", "transliteration": "khamsa"},
            {"number": 6, "arabic": "ستة", "transliteration": "sitta"},
            {"number": 7, "arabic": "سبعة", "transliteration": "sab'a"},
            {"number": 8, "arabic": "ثمانية", "transliteration": "thamaniya"},
            {"number": 9, "arabic": "تسعة", "transliteration": "tis'a"},
            {"number": 10, "arabic": "عشرة", "transliteration": "'ashara"}
        ]
    }
    return {"numbers": numbers}


@app.get("/api/pronouns", tags=["Grammar"])
async def get_pronouns():
    """Get Arabic pronouns (personal, possessive, demonstrative)"""
    if "pronouns" not in knowledge_base:
        raise HTTPException(status_code=404, detail="Pronouns data not found")
    
    return {
        "pronouns": knowledge_base["pronouns"]
    }


@app.get("/api/resources", tags=["General"])
async def get_learning_resources():
    """
    Get additional learning resources and references.
    
    Includes:
    - Tajweed rules for Quran recitation
    - Letter forms and positions (Markaz)
    - Recommended learning materials
    """
    return {
        "resources": [
            {
                "title": "Tajweed Rules",
                "description": "Rules for proper Quran recitation",
                "file": "arabic_tawjeed_rules.md",
                "topics": ["Makhraj", "Sifaat", "Idgham", "Ikhfa", "Qalqalah", "Ghunnah"]
            },
            {
                "title": "Letter Forms (Markaz)",
                "description": "Understanding letter positions in words",
                "file": "arabic_markaz_rules.md",
                "topics": ["Isolated", "Initial", "Medial", "Final"]
            },
            {
                "title": "Grammar Foundation",
                "description": "Core grammar concepts for beginners",
                "topics": ["Nouns", "Verbs", "Adjectives", "Sentence Structure"]
            }
        ],
        "external_links": [
            {
                "title": "Arabic Alphabet Guide",
                "url": "https://en.wikipedia.org/wiki/Arabic_alphabet",
                "description": "Comprehensive guide to Arabic letters"
            },
            {
                "title": "Quran.com",
                "url": "https://quran.com",
                "description": "Read and listen to Quran recitation"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
