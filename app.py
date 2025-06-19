from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import FinanceChatbot
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
chatbot = FinanceChatbot()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response = chatbot.chat(question)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 