import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_intent_model():
    # Cargar intents actualizados
    with open('data/intents.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Preparar datos
    texts = []
    labels = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            texts.append(pattern.lower())
            labels.append(intent['tag'])
    
    # Vectorización
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words=['de', 'y', 'en', 'la', 'el', 'que', 'con']
    )
    
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Guardar modelo y vectorizador
    with open('models/intent_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"✅ Modelo entrenado con {len(texts)} ejemplos y {len(set(labels))} intenciones")
    print(f"📊 Intenciones: {', '.join(set(labels))}")
    
    # Evaluación rápida
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    score = model.score(X_test, y_test)
    print(f"🎯 Precisión: {score:.2%}")

if __name__ == '__main__':
    train_intent_model()