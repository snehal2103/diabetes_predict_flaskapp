# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the trained model                                  
filename = 'C:/Users/SNEHAL/Desktop/diabetes_prediction/rtfe.sav'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        # Create a DataFrame with the correct feature names matching those used during training
        data = pd.DataFrame([[preg, glucose, bp, st, insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Make prediction
        my_prediction = classifier.predict(data)
        
        # Render the result in the result.html template
        return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
