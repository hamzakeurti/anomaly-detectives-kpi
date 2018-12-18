from predictors.PredictorTemplate import Predictor
import pandas as pd
import numpy as np

class MovingAveragePredictor(Predictor):
    def __init__(self, width, sigma):
        self.width = width
        self.sigma = sigma
    
    def fit(self):
        return

    def predict(self, data):
        values = data.value.values
        
        window = np.ones(self.width) / self.width
        avgs = np.convolve(values, window, 'same')
        devs = values - avgs
        std = np.std(devs)
        
        return pd.Series(abs(devs) > std * self.sigma)