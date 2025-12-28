from flask import Flask, render_template, request, jsonify
from chatbot.nl_engine import NLEngine
import traceback

app = Flask(__name__)

# Inicializar motor NLP
print("=" * 50)
print("🚀 INICIANDO CHATBOT PCO COMPUTACIÓN")
print("=" * 50)

try:
    nl_engine = NLEngine()
    print("✅ Motor NLP inicializado exitosamente")
except Exception as e:
    print(f"❌ Error crítico al inicializar NLP Engine: {e}")
    traceback.print_exc()
    nl_engine = None

@app.route('/')
def home():
    return render_template('index.html',
                         title="PCO Computación - Asistente Virtual",
                         company="PCO Computación",
                         slogan="Venta de equipos de cómputo e impresoras",
                         website="https://pco.com.mx")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Obtener mensaje del usuario
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Por favor envía un mensaje válido.',
                'confidence': 0.0,
                'intent': 'error',
                'timestamp': ''
            }), 400
        
        user_message = data['message']
        print(f"📩 Mensaje recibido: {user_message}")
        
        if nl_engine is None:
            return jsonify({
                'response': 'El asistente no está disponible temporalmente. Por favor intenta más tarde.',
                'confidence': 0.0,
                'intent': 'error',
                'timestamp': ''
            }), 503
        
        # Procesar con NLP
        result = nl_engine.process_query(user_message)
        
        print(f"✅ Respuesta generada: {result['intent']} (confianza: {result['confidence']:.2%})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"🔥 Error en endpoint /chat: {e}")
        traceback.print_exc()
        
        return jsonify({
            'response': 'Disculpa, hubo un error técnico. ¿En qué puedo ayudarte con equipos de cómputo?',
            'confidence': 0.0,
            'intent': 'error',
            'timestamp': ''
        }), 500

if __name__ == '__main__':
    print("\n🌐 Servidor listo en: http://localhost:5000")
    print("🔄 Presiona Ctrl+C para detener")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)