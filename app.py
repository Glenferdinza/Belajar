from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Aktifkan CORS agar frontend bisa akses API

# Load model dengan path relatif (penting untuk Heroku)
model_path = os.path.join(os.path.dirname(__file__), "model/car_price_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "model/scaler.pkl")

model = None
scaler = None

try:
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Cek apakah model punya method `predict()`
    if not hasattr(model, "predict"):
        raise ValueError("File model yang dimuat bukan model regresi yang valid.")
    
    # Cek apakah scaler punya method `transform()`
    if not hasattr(scaler, "transform"):
        raise ValueError("File scaler yang dimuat bukan scaler yang valid.")

except Exception as e:
    print(f"Error saat load model atau scaler: {e}")
    model, scaler = None, None  # Biar tidak crash kalau gagal load

@app.route("/")
def home():
    return "API Prediksi Harga Mobil"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model atau scaler gagal dimuat."}), 500

        data = request.get_json()

        # Pastikan semua input tersedia
        required_fields = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Data tidak lengkap"}), 400

        # Ambil data dari request JSON
        input_features = np.array([[ 
            float(data["Year"]),
            float(data["Engine_Size"]),
            float(data["Mileage"]),
            int(data["Doors"]),
            int(data["Owner_Count"])
        ]])

        # Normalisasi input agar sesuai dengan data training
        input_scaled = scaler.transform(input_features)

        # Prediksi harga
        predicted_price = model.predict(input_scaled)[0]

        # Format harga jadi dollar AS
        formatted_price = f"${predicted_price:,.2f}"

        return jsonify({"predicted_price": formatted_price})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Port Heroku secara otomatis akan disediakan di variabel lingkungan PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)