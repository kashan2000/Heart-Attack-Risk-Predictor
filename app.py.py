#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
model = pickle.load(open('heart.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        exang = int(request.form['exang'])
        caa = int(request.form['caa'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form['fbs'])
        rest_ecg = float(request.form['rest_ecg'])
        thalach = float(request.form['thalach'])
        cp= int(request.form['cp'])
        
        data = np.array([[sex, exang, caa,trtbps, chol,fbs,rest_ecg,thalach, age,cp]])
        my_prediction = model.predict(data)
        
        return render_template('Result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:




