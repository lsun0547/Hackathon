from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)


# Load your trained model (Random Forest, for example)
model = joblib.load('ghost_model.pkl')  # Make sure this path is correct


@app.route('/')
def home():
   return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   try:
       # Read input values from form
       features = [
           int(request.form['time']),
           int(request.form['location']),
           int(request.form['weather']),
           int(request.form['moon']),
           int(request.form['temperature']),
           int(request.form['day']),
           int(request.form['wind'])
       ]


       prediction = model.predict([features])[0]
       result = "Haunted" if prediction == 1 else "Not Haunted"


       return f"<h2>Prediction: {result}</h2><a href='/'>Back</a>"


   except Exception as e:
       return f"<h2>Error: {str(e)}</h2><a href='/'>Back</a>"


if __name__ == '__main__':
   app.run(debug=True)