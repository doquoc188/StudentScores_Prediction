import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_preprocessing import (
    load_data,
    prepare_data,
    FeatureEngineer,
    EDUCATION_ORDER
)

class TestPreprocessing(unittest.TestCase):

    def test_load_data(self):
        df = load_data("StudentScore.xls")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("math score", df.columns)
        self.assertIn("reading score", df.columns)
        self.assertIn("writing score", df.columns)

    def test_feature_engineer(self):
        data = pd.DataFrame({
            "reading score": [80.0, 60.0],
            "writing score": [70.0, 60.0]
        })
        fe = FeatureEngineer()
        res = fe.transform(data)
        
        self.assertIn("reading_writing_avg", res.columns)
        self.assertIn("reading_writing_diff", res.columns)
        self.assertIn("reading_writing_ratio", res.columns)
        self.assertEqual(res["reading_writing_avg"].tolist(), [75.0, 60.0])
        self.assertEqual(res["reading_writing_diff"].tolist(), [10.0, 0.0])

    def test_prepare_data_split(self):
        X_train, X_test, y_train, y_test = prepare_data("StudentScore.xls")
        self.assertEqual(len(X_train), 800)
        self.assertEqual(len(X_test), 200)
        self.assertEqual(len(y_train), 800)
        self.assertEqual(len(y_test), 200)
        self.assertNotIn("math score", X_train.columns)

if __name__ == "__main__":
    unittest.main()
