from sklearn.ensemble import RandomForestClassifier

from predictors.PredictorTemplate import Predictor


class RandomForestPredictor(Predictor):

    def __init__(self):
        self.clf = RandomForestClassifier(n_jobs=2, random_state=0)

    def fit(self, X, Y):
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)
