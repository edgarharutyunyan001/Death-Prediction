import pandas as pd
from sklearn.impute import KNNImputer
import joblib


class Preprocessor():
    def __init__(self, path):
        self.path = path
        self.drop_cols = ['recordid', 'Survival', 'Length_of_stay', 'SAPS-I', 'SOFA']
        self.data = pd.read_csv(path)
        self.data.drop(self.drop_cols, axis=1, inplace=True)

    def fit(self):
        label = self.data['In-hospital_death']
        self.data.drop("In-hospital_death", axis=1, inplace=True)
        imputer = KNNImputer()
        imputer.fit(self.data)
        X = pd.DataFrame(imputer.transform(self.data), columns=self.data.columns)
        # Save the imputer
        joblib.dump(imputer, 'trained_imputer.joblib')
        return X, label

    def transform(self):
        # Load necessary model
        self.data.drop("In-hospital_death", axis=1, inplace=True)
        imputer = joblib.load('trained_imputer.joblib')
        # Fill nan values in data
        X = pd.DataFrame(imputer.transform(self.data), columns=self.data.columns)
        return X