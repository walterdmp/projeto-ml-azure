import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app) # Permite conexão com o React

# Carregar modelo de forma segura para nuvem
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.pkl')

try:
    model = pickle.load(open(model_path, "rb"))
except:
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API Online Azure", "msg": "Use POST em /api/predict"})

@app.route("/api/predict", methods=["POST"])
def results():
    if not model:
        return jsonify({"erro": "Modelo não carregado"}), 500
    
    data = request.get_json(force=True)
    vals = [float(data['ano']), float(data['km']), float(data['motor'])]
    prediction = model.predict([np.array(vals)])
    
    return jsonify({"preco_estimado": f"R$ {prediction[0]:,.2f}"})

if __name__ == "__main__":
    app.run(debug=True)