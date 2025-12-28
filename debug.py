import sys
import os

print("=== DIAGNÓSTICO DEL CHATBOT ===")
print(f"Python: {sys.version}")
print(f"Directorio actual: {os.getcwd()}")

# Verificar archivos
files_to_check = [
    'app.py',
    'chatbot/nl_engine.py',
    'models/intent_classifier.pkl',
    'models/tfidf_vectorizer.pkl',
    'data/intents.json'
]

for file in files_to_check:
    exists = os.path.exists(file)
    print(f"{'✅' if exists else '❌'} {file}: {'EXISTE' if exists else 'NO EXISTE'}")

# Verificar imports
print("\n=== PRUEBA DE IMPORTS ===")
try:
    from chatbot.nl_engine import NLEngine
    print("✅ Import de NLEngine exitoso")
    
    engine = NLEngine()
    print("✅ Instancia de NLEngine creada")
    
    # Probar con un mensaje
    result = engine.process_query("hola")
    print(f"✅ Prueba de mensaje exitosa:")
    print(f"   Respuesta: {result['response'][:50]}...")
    print(f"   Confianza: {result['confidence']}")
    print(f"   Intención: {result['intent']}")
    
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()