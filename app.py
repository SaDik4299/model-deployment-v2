from flask import Flask, render_template, request
import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=None, email="")

@app.route("/predict", methods=['POST'])
def predict():
    email = ""
    prediction = None
    
    if request.method == 'POST':
        email = request.form.get("content", "")
        
        if email.strip():
            tokenized_email = cv.transform([email])
            raw_prediction = model.predict(tokenized_email)
            prediction = 1 if raw_prediction == 1 else -1

    return render_template("index.html", prediction=prediction, email=email)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
