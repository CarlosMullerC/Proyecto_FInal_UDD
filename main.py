from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Cargar el modelo y el scaler
model_path = r'C:\Users\Carlos\Documents\GitHub\Proyecto_Final_UDD\best_models.pkl'
scaler_path = r'C:\Users\Carlos\Documents\GitHub\Proyecto_Final_UDD\scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def prepare_data(data):
    df = pd.DataFrame([data])  # Convertir el diccionario en un DataFrame con una fila
    
    # Aplicar estandarización
    standardized_data = scaler.transform(df)
    
    return standardized_data

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Datos recibidos:", data)  # Depuración: Imprimir los datos recibidos

        # Verificar y preparar los datos
        if 'data_values' not in data:
            raise KeyError("La clave 'data_values' no está presente en los datos de entrada.")
        
        data_values = data['data_values']
        prepared_data = prepare_data(data_values)
        print("Datos preparados:", prepared_data)  # Para depuración
        
        # Realiza predicciones con el modelo
        prediction = model.predict(prepared_data)
        
        # Devuelve la respuesta
        response = {"output_data": prediction.tolist()}
        return jsonify(response)
    except KeyError as ke:
        return jsonify({"error": str(ke)}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)