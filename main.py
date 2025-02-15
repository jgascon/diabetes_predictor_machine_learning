#!/usr/bin/env python3
#
#  main.py
#
#  Copyright 2025 Unknown <jorge@gremlin>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import sys
import kagglehub

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from flask import Flask, render_template, request



def download_diabetes_dataset():
    # Download diabetes data:
    diabetes_dataset_path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    print("Diabetes dataset downloaded to: ["+diabetes_dataset_path+"]")
    diabetes_dataset_path += '/diabetes.csv'
    #/home/<YOUR_USER>/.cache/kagglehub/datasets/uciml/pima-indians-diabetes-database/versions/1/diabetes.csv
    return diabetes_dataset_path



class DiabetesPredictor:
    def __init__(self):
        self.diabetes_dataset_filepath = download_diabetes_dataset()
        self.dataset = pd.read_csv(self.diabetes_dataset_filepath)
        print('self.dataset.head:\n',self.dataset.head(5))
        print('self.dataset.shape:\n',self.dataset.shape)
        print('self.dataset.describe:\n',self.dataset.describe())
        features = self.dataset.drop(['Outcome'], axis=1)
        labels = self.dataset['Outcome']
        self.features_train, self.features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)
        self.classifier = KNeighborsClassifier()
        self.classifier.fit(self.features_train, labels_train)
        pred = self.classifier.predict(self.features_test)
        self.accuracy = accuracy_score(labels_test, pred)
        print('Accuracy: {}'.format(self.accuracy))
        print('Train size: {}'.format(len(self.features_train)))
        print('Test size: {}'.format(len(self.features_test)))

    def predict(self, data):
        # ~ data = {'Pregnancies': [1],
                # ~ 'Glucose': [85],
                # ~ 'BloodPressure': [66],
                # ~ 'SkinThickness': [29],
                # ~ 'Insulin': [0],
                # ~ 'BMI': [26.6],
                # ~ 'DiabetesPedigreeFunction': [0.351],
                # ~ 'Age': [31]
        # ~ }
        df = pd.DataFrame(data)
        print(df)
        pred = self.classifier.predict(df)
        return pred[0]



app = Flask(__name__)
DIABETES_PREDICTOR = None

@app.route("/")
def welcome_page():
    return render_template("file1.html")

@app.route("/file2", methods=['POST'])
def second_page():
    # ~ html_data = request.form["enter_value"]
    diabetes_data = {}
    diabetes_data['Pregnancies'] = [request.form['Pregnancies']]
    diabetes_data['Glucose'] = [request.form['Glucose']]
    diabetes_data['BloodPressure'] = [request.form['BloodPressure']]
    diabetes_data['SkinThickness'] = [request.form['SkinThickness']]
    diabetes_data['Insulin'] = [request.form['Insulin']]
    diabetes_data['BMI'] = [request.form['BMI']]
    diabetes_data['DiabetesPedigreeFunction'] = [request.form['DiabetesPedigreeFunction']]
    diabetes_data['Age'] = [request.form['Age']]
    result = DIABETES_PREDICTOR.predict(diabetes_data)
    if result == 0:
        diabetes_result = "Negative"
    else:
        diabetes_result = "Positive"
    return render_template("file2.html",
        diabetes_result = diabetes_result,
        number_samples = len(DIABETES_PREDICTOR.features_train),
        accuracy  = DIABETES_PREDICTOR.accuracy
    )


if __name__== '__main__':
    DIABETES_PREDICTOR = DiabetesPredictor()
    app.run(host='0.0.0.0', debug=True, port=5000)

