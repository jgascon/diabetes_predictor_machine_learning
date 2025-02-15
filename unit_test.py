import unittest
from pathlib import Path

import main

class TestDiabetesPredictor(unittest.TestCase):
    def test_download_dataset(self):
        path = main.download_diabetes_dataset()
        #/home/<YOUR_USER>/.cache/kagglehub/datasets/uciml/pima-indians-diabetes-database/versions/1/diabetes.csv
        home_path = Path.home()._str
        expected_path = home_path + '/.cache/kagglehub/datasets/uciml/pima-indians-diabetes-database/versions/1/diabetes.csv'
        self.assertEqual(path == expected_path, True)
        my_file = Path(path)
        self.assertEqual(my_file.is_file(), True)

    def test_diabetes_predictor(self):
        predictor = main.DiabetesPredictor()
        data = {'Pregnancies': [1],
                'Glucose': [85],
                'BloodPressure': [66],
                'SkinThickness': [29],
                'Insulin': [0],
                'BMI': [26.6],
                'DiabetesPedigreeFunction': [0.351],
                'Age': [31]
        }
        result = predictor.predict(data)
        self.assertEqual(result == 0, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
