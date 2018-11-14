from sklearn.ensemble import RandomForestClassifier


class RandomForestPredictor:

    def train(self, X, Y):
        self.clf = RandomForestClassifier(n_jobs=2, random_state=0)
        self.clf.fit(X, Y)

    def predict(self, X):
        return self.clf.predict(X)
