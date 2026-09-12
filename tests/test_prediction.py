import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.predict import StudentScorePredictor

class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = StudentScorePredictor()

    def test_predictor_single(self):
        sample_input = {
            "gender": "female",
            "race/ethnicity": "group B",
            "parental level of education": "bachelor's degree",
            "lunch": "standard",
            "test preparation course": "none",
            "reading score": 72.0,
            "writing score": 74.0
        }
        
        result = self.predictor.predict_single(sample_input)
        self.assertIn("predicted_math_score", result)
        self.assertIn("rounded_math_score", result)
        self.assertTrue(0 <= result["predicted_math_score"] <= 100)
        self.assertTrue(0 <= result["rounded_math_score"] <= 100)

    def test_predictor_batch(self):
        sample_df = pd.DataFrame([
            {
                "gender": "female",
                "race/ethnicity": "group B",
                "parental level of education": "bachelor's degree",
                "lunch": "standard",
                "test preparation course": "none",
                "reading score": 72.0,
                "writing score": 74.0
            },
            {
                "gender": "male",
                "race/ethnicity": "group C",
                "parental level of education": "master's degree",
                "lunch": "free/reduced",
                "test preparation course": "completed",
                "reading score": 85.0,
                "writing score": 90.0
            }
        ])
        
        df_out = self.predictor.predict_batch(sample_df)
        self.assertIn("predicted_math_score", df_out.columns)
        self.assertEqual(len(df_out), 2)

if __name__ == "__main__":
    unittest.main()
