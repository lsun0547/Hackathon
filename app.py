import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('ghost_model.pkl')  # load model that we have trained using fake data

@app.route('/')
def home():
    return render_template('index.html')  # render html page for our input ui

@app.route('/predict', methods=['POST'])  # handle form submission from rendered html page
def predict():
    try:
        # read input values from form
        features = [
            int(request.form['time']),
            int(request.form['location']),
            int(request.form['weather']),
            int(request.form['moon']),
            int(request.form['temperature']),
            int(request.form['day'])
        ]

        prediction = model.predict([features])[0]
        result = "Haunted" if prediction == 1 else "Not Haunted"  # based on parameters from user, use ml model to predict if graveyard will be haunted or not
        bg_color = "black" if prediction == 1 else "gray"

        return render_template('prediction.html', result=result, bg_color=bg_color)  # display prediction result

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
