from flask import Flask, request, render_template
import numpy as np
import pandas as pd

import os, sys

from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            writing_score = float(request.form.get('writing_score')),
            reading_score = float(request.form.get('reading_score')),
        )

        pred_df = data.get_data_as_frame()
        print (pred_df)

        predict_pipepline = PredictPipeline()

        results = predict_pipepline.predict(pred_df)

        print("prediction is ")
        print(results[0])

        return render_template('home.html', prediction = results[0])
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

