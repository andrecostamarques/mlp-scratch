import sys
from pathlib import Path

# Adiciona a pasta PAI (root do projeto) ao sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request, jsonify
import numpy as np
from NMLPlib import *

app = Flask(__name__)

# Carrega o modelo uma vez na inicialização
modelo = loadModelo('../NeuralNetwork.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== DEBUG: Iniciando predição ===")
        
        # Recebe a matriz 28x28 do frontend
        data = request.json
        print(f"DEBUG: Dados recebidos: {type(data)}")
        
        grid_data = data['grid']
        print(f"DEBUG: Grid shape: {len(grid_data)}x{len(grid_data[0]) if grid_data else 0}")
        
        # Converte a matriz para numpy array 28x28
        img_array = np.array(grid_data, dtype=np.uint8)
        print(f"DEBUG: Array shape: {img_array.shape}")
        print(f"DEBUG: Array dtype: {img_array.dtype}")
        print(f"DEBUG: Array min/max: {img_array.min()}/{img_array.max()}")
        
        # Faz a inferência
        print("DEBUG: Chamando modelo...")
        numero, certeza = modelo.preverImageCustomOnlyText(img_array)
        print(f"DEBUG: Resultado - Número: {numero}, Certeza: {certeza}")
        
        return jsonify({
            'success': True,
            'numero': int(numero),
            'certeza': float(certeza)
        })
        
    except Exception as e:
        print(f"ERRO DETALHADO: {str(e)}")
        print(f"TIPO DO ERRO: {type(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/clear', methods=['POST'])
def clear():
    return jsonify({'success': True, 'message': 'Canvas cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)