import os
import time
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

import torch
import tiktoken
from model_def import GPTModel, BASE_CONFIG

# Initialize and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load('review_classifier.pth', map_location=device))
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
max_length = 120
pad_token_id = 50256

def predict_spam(text):
    """
    Actual model inference code.
    """
    try:
        # Prepare inputs to the model
        input_ids = tokenizer.encode(text)
        supported_context_length = model.pos_emb.weight.shape[0]
        
        # Truncate sequences if they are too long
        input_ids = input_ids[:min(max_length, supported_context_length)]
        
        # Pad sequences to the longest sequence
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension
        
        # Model inference
        with torch.no_grad():
            logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
            probas = torch.softmax(logits, dim=-1)
            
        predicted_label = torch.argmax(probas, dim=-1).item()
        confidence = probas[0][predicted_label].item()
        
        return {
            "prediction": "Spam" if predicted_label == 1 else "Ham (Not Spam)", 
            "confidence": confidence
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"prediction": "Error", "confidence": 0.0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
            
        # Get prediction from model
        result = predict_spam(message)
        
        return jsonify({
            'success': True,
            'message': message,
            'prediction': result['prediction'],
            'confidence': result['confidence']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Using debug=True for development. Disable in production.
    app.run(debug=True, host='0.0.0.0', port=5000)
