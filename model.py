from sklearn.ensemble import RandomForestClassifier
import joblib


class Model():
    def __init__(self):
        pass

    def fit(self, data, y):
        weight = sum(y == 0)/sum(y == 1)
        model = RandomForestClassifier(class_weight={0: 1, 1: weight}, min_samples_split=5)
        model.fit(data, y)
        joblib.dump(model, 'Model.joblib')

    def predict(self, X):
        model = joblib.load('Model.joblib')
        return model.predict_proba(X)[:, 1]