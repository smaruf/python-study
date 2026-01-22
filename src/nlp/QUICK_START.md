# Quick Start Guide - Arabic Learning Platform

## 🚀 Getting Started in 5 Minutes

This guide will help you set up and run the Arabic Learning Platform quickly.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for installing packages)

### Step 1: Install Dependencies

Navigate to the NLP directory:

```bash
cd /path/to/python-study/src/nlp
```

Install required packages:

```bash
pip install -r nlp_requirements.txt
```

**Note**: If you encounter any issues, you can install packages individually:

```bash
pip install fastapi uvicorn pydantic tensorflow scikit-learn joblib numpy
```

### Step 2: Prepare the Model (First Time Only)

Before using the API, you need to train the machine learning model:

```bash
python dynamic_arabic_grammar.py
```

**Expected Output:**
```
Loading existing model...
# OR
Training a new model...
Epoch 1/10
...
System is ready! Start asking questions.
```

This creates:
- `saved_tensorflow_model/` - The trained AI model
- `vectorizer.joblib` - Text processing component
- `label_encoder.joblib` - Classification component
- `processed_knowledge_base.json` - Combined data

**Time:** ~2-5 minutes depending on your hardware

### Step 3: Start the API Server

```bash
python arabic_learning_api.py
```

**Alternative (recommended for development):**
```bash
uvicorn arabic_learning_api:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
Arabic Learning API started successfully!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Test the API

Open your browser and visit:

**Interactive Documentation:**
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

**Or use the command line:**

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Arabic letter ب?"}'

# Get vocabulary
curl "http://localhost:8000/api/vocabulary?category=verbs&limit=5"

# Get conversation scenarios
curl "http://localhost:8000/api/conversations?difficulty=beginner"
```

### Step 5: Try the Python Client Examples

In a new terminal (keep the server running):

```bash
# Install requests if needed
pip install requests

# Run examples
python api_client_examples.py
```

**Expected Output:**
```
🌙 Arabic Learning API - Client Examples

============================================================
  1. Health Check
============================================================
Status: healthy
Models loaded: {'vectorizer': True, 'label_encoder': True, ...}

============================================================
  2. Ask Questions
============================================================
Q: What is the Arabic letter ب?
A: Ba - Sound: B - Example: بيت (Bayt - House)
Confidence: 95.23%
...
```

## 🖥️ Alternative: Desktop GUI Application

If you prefer a standalone desktop application:

```bash
python arabic_grammar_gui.py
```

This opens a graphical interface where you can:
- Ask questions about Arabic grammar
- Get instant answers from the AI model
- Learn interactively without coding

## 📱 Building a Mobile/Web App

### React Native Example

```javascript
// Install: npm install axios
import axios from 'axios';

const API_BASE = 'http://your-server-ip:8000';

// Get vocabulary
const getVocabulary = async () => {
  const response = await axios.get(`${API_BASE}/api/vocabulary`, {
    params: { category: 'nouns', limit: 20 }
  });
  return response.data.items;
};

// Ask a question
const askQuestion = async (question) => {
  const response = await axios.post(`${API_BASE}/api/query`, {
    question: question
  });
  return response.data.answer;
};

// Usage in component
const MyComponent = () => {
  const [answer, setAnswer] = useState('');
  
  const handleAsk = async () => {
    const result = await askQuestion('What does السلام عليكم mean?');
    setAnswer(result);
  };
  
  return <Text>{answer}</Text>;
};
```

### Flutter Example

```dart
// Add to pubspec.yaml: http: ^1.1.0
import 'package:http/http.dart' as http;
import 'dart:convert';

class ArabicLearningAPI {
  static const String baseUrl = 'http://your-server-ip:8000';
  
  static Future<List> getConversations(String difficulty) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/conversations?difficulty=$difficulty')
    );
    
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return data['scenarios'];
    }
    throw Exception('Failed to load conversations');
  }
  
  static Future<Map> askQuestion(String question) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/query'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'question': question}),
    );
    
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Failed to get answer');
  }
}
```

## 🐛 Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install --upgrade pip
pip install -r nlp_requirements.txt
```

### Issue: "Model not found" error when starting API

**Solution:**
```bash
# Train the model first
python dynamic_arabic_grammar.py
```

### Issue: Cannot connect to API from mobile device

**Solution:**
1. Make sure the server is running: `uvicorn arabic_learning_api:app --host 0.0.0.0 --port 8000`
2. Find your computer's IP address:
   - **Windows**: `ipconfig` (look for IPv4)
   - **Mac/Linux**: `ifconfig` or `ip addr` (look for inet)
3. Use that IP in your mobile app: `http://192.168.1.XXX:8000`
4. Make sure firewall allows port 8000

### Issue: CORS errors in web browser

**Solution:**
The API has CORS enabled by default. If you still get errors, you can modify the CORS settings in `arabic_learning_api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Slow API responses

**Solution:**
- First request is always slower (model loading)
- Subsequent requests should be fast
- Consider using a more powerful server for production
- For very large scale, look into model optimization and caching

## 📊 API Endpoints Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and endpoint list |
| `/api/health` | GET | Health check and status |
| `/api/query` | POST | Ask questions about Arabic |
| `/api/vocabulary` | GET | Get vocabulary by category |
| `/api/alphabet` | GET | Get Arabic alphabet |
| `/api/conversations` | GET | Get conversation scenarios |
| `/api/reading/passages` | GET | Get reading exercises |
| `/api/grammar/rules` | GET | Get grammar rules |
| `/api/numbers` | GET | Get Arabic numbers |
| `/api/pronouns` | GET | Get Arabic pronouns |
| `/docs` | GET | Interactive API documentation |

## 📚 Learning Resources

### For Beginners
1. Start with `/api/alphabet` to learn letters
2. Practice greetings from `/api/conversations?difficulty=beginner`
3. Read simple passages from `/api/reading/passages?difficulty=beginner`
4. Ask questions using `/api/query`

### For Intermediate Learners
1. Expand vocabulary with `/api/vocabulary`
2. Practice conversations: `/api/conversations?difficulty=intermediate`
3. Study grammar: `/api/grammar/rules`
4. Read longer passages: `/api/reading/passages?difficulty=intermediate`

### For Advanced Learners
1. Read complex texts: `/api/reading/passages?difficulty=advanced`
2. Practice advanced conversations
3. Study Tajweed rules in `arabic_tawjeed_rules.md`
4. Contribute new content to the platform!

## 🔐 Production Deployment

For deploying to production:

1. **Security**: Add authentication (JWT tokens)
2. **HTTPS**: Use SSL/TLS certificates
3. **CORS**: Restrict to specific origins
4. **Rate Limiting**: Prevent abuse
5. **Monitoring**: Add logging and monitoring
6. **Scaling**: Use gunicorn with multiple workers

Example production command:
```bash
gunicorn arabic_learning_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile -
```

## 🎯 Next Steps

1. ✅ Read the complete documentation: `LEARNING_GUIDE.md`
2. ✅ Explore the interactive API docs: http://localhost:8000/docs
3. ✅ Try the client examples: `python api_client_examples.py`
4. ✅ Build your own app using the API
5. ✅ Contribute new vocabulary and content

## 💡 Tips

- **Learn Systematically**: Start with alphabet, then basic words, then phrases, then sentences
- **Practice Daily**: Use the conversation scenarios every day
- **Read Regularly**: Try to read one passage per day
- **Ask Questions**: Use the query endpoint to clarify doubts
- **Track Progress**: Keep notes on what you've learned

## 🤝 Need Help?

- Check `LEARNING_GUIDE.md` for detailed documentation
- Visit http://localhost:8000/docs for API reference
- Open an issue on GitHub for bugs or questions

---

**Happy Learning! 🌙 بالتوفيق (Good Luck!)**
