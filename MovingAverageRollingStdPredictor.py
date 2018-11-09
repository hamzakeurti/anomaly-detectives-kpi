from MovingAveragePredictor import MovingAveragePredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MovingAverageRollingStdPredictor(MovingAveragePredictor):
    def predict(self, datas):
        predictions = {}
        i = 0
        for id in datas:
            data = datas[id]
            values = data['value']
            timestamps = data['timestamp']
            labels = data['label']
            
            window = np.ones(self.width) / self.width
            avgs = values.rolling(self.width, center=True).mean()
            std = values.rolling(self.width, center=True).std()
            devs = values - avgs
            predictions[id] = pd.Series(abs(devs) > std * self.sigma)
        return predictions