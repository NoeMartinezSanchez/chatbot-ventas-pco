import json
import pickle
import numpy as np
import random
from datetime import datetime

class NLEngine:
    def __init__(self):
        print("🤖 Inicializando ChatBot PCO...")
        
        # Cargar modelos
        try:
            with open('models/intent_classifier.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('data/intents.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.intents = data['intents']
            
            # Crear mapa de tags a respuestas
            self.responses = {}
            for intent in self.intents:
                self.responses[intent['tag']] = intent['responses']
            
            print(f"✅ Modelo cargado: {len(self.intents)} categorías")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.model = None
            self.vectorizer = None
            self.responses = {}
    
    def process_query(self, user_input: str):
        """Procesa la consulta del usuario"""
        try:
            if not self.model or not self.vectorizer:
                return self._get_fallback_response()
            
            # Preprocesar
            text = user_input.lower().strip()
            
            # Vectorizar
            X = self.vectorizer.transform([text])
            
            # Predecir
            tag = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = np.max(proba)
            
            # Obtener respuesta
            if tag in self.responses:
                response = random.choice(self.responses[tag])
            else:
                response = self._get_fallback_response()['response']
            
            return {
                'response': response,
                'confidence': float(confidence),
                'intent': tag,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error: {e}")
            return self._get_fallback_response()
    
    def _get_fallback_response(self):
        """Respuesta cuando hay error"""
        fallbacks = [
            "¡Hola! En PCO Computación ofrecemos equipos de cómputo e impresoras. ¿En qué puedo asistirte?",
            "¿Buscas laptops, desktops o impresoras? Tenemos las mejores marcas.",
            "Puedo ayudarte con información sobre productos, precios y envíos. ¿Qué necesitas?"
        ]
        
        return {
            'response': random.choice(fallbacks),
            'confidence': 0.5,
            'intent': 'saludo',
            'timestamp': datetime.now().isoformat()
        }