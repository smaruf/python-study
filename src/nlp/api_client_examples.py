"""
Arabic Learning API - Python Client Examples
============================================

This file demonstrates how to interact with the Arabic Learning API
using Python requests library.

Installation:
    pip install requests

Usage:
    python api_client_examples.py
"""

import requests
import json
from typing import Dict, List, Optional


class ArabicLearningClient:
    """Client class for interacting with the Arabic Learning API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: The base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check if the API is running and healthy"""
        response = self.session.get(f"{self.base_url}/api/health")
        response.raise_for_status()
        return response.json()
    
    def ask_question(self, question: str) -> Dict:
        """
        Ask a question about Arabic grammar or vocabulary
        
        Args:
            question: Your question in English or Arabic
            
        Returns:
            Dictionary with question, answer, and confidence score
        """
        response = self.session.post(
            f"{self.base_url}/api/query",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    
    def get_vocabulary(self, category: Optional[str] = None, limit: int = 20) -> Dict:
        """
        Get vocabulary items by category
        
        Args:
            category: 'nouns', 'verbs', or 'adjectives' (None for mixed)
            limit: Number of items to return
            
        Returns:
            Dictionary with vocabulary items
        """
        params = {"limit": limit}
        if category:
            params["category"] = category
        
        response = self.session.get(
            f"{self.base_url}/api/vocabulary",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_alphabet(self) -> Dict:
        """Get the complete Arabic alphabet with pronunciation"""
        response = self.session.get(f"{self.base_url}/api/alphabet")
        response.raise_for_status()
        return response.json()
    
    def get_conversations(self, difficulty: Optional[str] = None) -> Dict:
        """
        Get conversation practice scenarios
        
        Args:
            difficulty: 'beginner', 'intermediate', or 'advanced' (None for all)
            
        Returns:
            Dictionary with conversation scenarios
        """
        params = {}
        if difficulty:
            params["difficulty"] = difficulty
        
        response = self.session.get(
            f"{self.base_url}/api/conversations",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_reading_passages(self, difficulty: Optional[str] = None, limit: int = 5) -> Dict:
        """
        Get reading comprehension passages
        
        Args:
            difficulty: 'beginner', 'intermediate', or 'advanced' (None for all)
            limit: Number of passages to return
            
        Returns:
            Dictionary with reading passages
        """
        params = {"limit": limit}
        if difficulty:
            params["difficulty"] = difficulty
        
        response = self.session.get(
            f"{self.base_url}/api/reading/passages",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_grammar_rules(self, rule: Optional[str] = None) -> Dict:
        """
        Get Arabic grammar rules
        
        Args:
            rule: Specific rule name (None for all rules)
            
        Returns:
            Dictionary with grammar rules
        """
        params = {}
        if rule:
            params["rule"] = rule
        
        response = self.session.get(
            f"{self.base_url}/api/grammar/rules",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def get_numbers(self) -> Dict:
        """Get Arabic numbers with pronunciation"""
        response = self.session.get(f"{self.base_url}/api/numbers")
        response.raise_for_status()
        return response.json()
    
    def get_pronouns(self) -> Dict:
        """Get Arabic pronouns"""
        response = self.session.get(f"{self.base_url}/api/pronouns")
        response.raise_for_status()
        return response.json()


def print_section(title: str):
    """Print a formatted section title"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def main():
    """Main function demonstrating API usage"""
    
    # Initialize the client
    client = ArabicLearningClient()
    
    print("🌙 Arabic Learning API - Client Examples\n")
    
    try:
        # 1. Health Check
        print_section("1. Health Check")
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
        
        # 2. Ask a Question
        print_section("2. Ask Questions")
        questions = [
            "What is the Arabic letter ب?",
            "What does السلام عليكم mean?",
            "How do I say thank you in Arabic?"
        ]
        
        for q in questions:
            result = client.ask_question(q)
            print(f"\nQ: {q}")
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2%}")
        
        # 3. Get Vocabulary
        print_section("3. Vocabulary - Common Verbs")
        vocab = client.get_vocabulary(category="verbs", limit=5)
        print(f"Found {vocab['count']} verbs:")
        for item in vocab['items'][:5]:
            print(f"  {item['word']} - {item['meaning']}")
            if 'example' in item:
                print(f"    Example: {item['example']}")
        
        # 4. Get Alphabet
        print_section("4. Arabic Alphabet (First 5 letters)")
        alphabet = client.get_alphabet()
        count = 0
        for letter, details in alphabet['letters'].items():
            if count >= 5:
                break
            print(f"  {letter} - {details['name']} (Sound: {details['sound']})")
            print(f"    Example: {details['example']}")
            count += 1
        
        # 5. Get Conversation Scenarios
        print_section("5. Conversation Practice - Beginner")
        conversations = client.get_conversations(difficulty="beginner")
        if conversations['scenarios']:
            scenario = conversations['scenarios'][0]
            print(f"\nScenario: {scenario['scenario']}")
            print(f"Difficulty: {scenario['difficulty']}")
            print("\nSample phrases:")
            for phrase in scenario['phrases'][:3]:
                print(f"  Arabic: {phrase['arabic']}")
                print(f"  English: {phrase['english']}")
                print(f"  Pronunciation: {phrase['transliteration']}\n")
        
        # 6. Get Reading Passages
        print_section("6. Reading Comprehension - Beginner")
        passages = client.get_reading_passages(difficulty="beginner", limit=1)
        if passages['passages']:
            passage = passages['passages'][0]
            print(f"\nTitle: {passage['title']}")
            print(f"Difficulty: {passage['difficulty']}")
            print(f"\nArabic Text:\n{passage['text_arabic']}")
            print(f"\nEnglish Translation:\n{passage['text_english']}")
            print(f"\nVocabulary ({len(passage['vocabulary'])} words):")
            for word in passage['vocabulary'][:5]:
                print(f"  {word['word']} - {word['meaning']}")
        
        # 7. Get Numbers
        print_section("7. Arabic Numbers (1-10)")
        numbers = client.get_numbers()
        if 'numbers' in numbers and 'cardinal' in numbers['numbers']:
            for num in numbers['numbers']['cardinal'][:10]:
                print(f"  {num['number']}: {num['arabic']} ({num['transliteration']})")
        
        # 8. Grammar Rules
        print_section("8. Grammar Rules Sample")
        rules = client.get_grammar_rules()
        if 'rules' in rules:
            # Show first 3 rules
            count = 0
            for rule_name, explanation in rules['rules'].items():
                if count >= 3:
                    break
                print(f"\n{rule_name}:")
                print(f"  {explanation}")
                count += 1
        
        print_section("Examples Complete!")
        print("\n✅ All API calls successful!")
        print("\n📚 For more information, visit: http://localhost:8000/docs")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API server.")
        print("Make sure the server is running with:")
        print("  python arabic_learning_api.py")
        print("Or:")
        print("  uvicorn arabic_learning_api:app --reload")
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
