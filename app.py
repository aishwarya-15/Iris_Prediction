from flask import Flask,request,render_template
from flask_wtf import FlaskForm
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(final)
    prediction = model.predict(final)
    return render_template('index.html',res = str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
