import unittest
from joblib import load
import pandas as pd

class TestModel(unittest.TestCase):
    model=None
    model_path="artifacts/model.joblib"
    ds= pd.read_csv('data/iris.csv')
    ds['sample_id'] = ds.index
    cols = ['sample_id', 'sepal_length', 'petal_length', 'sepal_width', 'petal_width', 'species']
    ds = ds[cols]
    print("Enering into the test module.")
    print(ds.head())
    
    def setUp(self):
        self.model=load(self.model_path)
        
    def test_data_validation(self):
        # Ensure no missing values
        self.assertFalse(self.ds.drop(columns='species').isnull().values.any(), "Missing values in features")
        self.assertFalse(self.ds['species'].isnull().values.any(), "Missing values in labels")

        # Ensure correct number of features
        self.assertEqual(self.ds.drop(columns='species').shape[1], 5, "Expected 5 features")
        
    def test_evaluation(self):
        sample = self.ds.drop(columns=['species'])
        print(sample)
        result = self.model.predict(sample[sorted(sample.columns)].iloc[0:1])
        print(result)
        expected = self.ds['species'].iloc[0]
        self.assertEqual(result[0], expected, "Wrong prediction")
            
if __name__=="__main__":
    unittest.main()
