from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/crop_model.pkl", "rb"))

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = [
        float(request.form['N']),
        float(request.form['P']),
        float(request.form['K']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]

    final_input = np.array([data])
    prediction = model.predict(final_input)[0]

    return render_template(
        "index.html",
        prediction_text=f"Recommended Crop: {prediction}"
    )

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

