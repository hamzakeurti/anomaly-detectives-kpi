from Predictor import Predictor
import pandas as pd
import numpy as np

class MovingAveragePredictor(Predictor):
    def __init__(self, width, sigma):
        self.width = width
        self.sigma = sigma
    
    def train(self):
        return

    def predict(self, datas):
        predictions = {}
        i = 0
        for id in datas:
            data = datas[id]
            values = data['value']
            timestamps = data['timestamp']
            labels = data['label']
            
            window = np.ones(self.width) / self.width
            avgs = np.convolve(values, window, 'same')
            devs = values - avgs
            std = np.std(devs)
            
            predictions[id] = pd.Series(abs(devs) > std * self.sigma)
        return predictions