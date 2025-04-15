# app/app.py
from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('gold_price_model.pkl')

@app.route('/')
def home():
    with open('index.html', 'r') as file:
        return render_template_string(file.read())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        spx = float(request.form['spx'])
        uso = float(request.form['uso'])
        slv = float(request.form['slv'])
        eurusd = float(request.form['eurusd'])

        features = np.array([[spx, uso, slv, eurusd]])
        prediction = model.predict(features)[0]
        return f"<h2>Predicted Gold Price: ${prediction:.2f}</h2><br><a href='/'>Try Again</a>"
    except Exception as e:
        return f"<h2>Error: {str(e)}</h2><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
