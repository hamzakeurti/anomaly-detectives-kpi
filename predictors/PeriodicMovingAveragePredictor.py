from predictors.PredictorTemplate import Predictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization.visualize import visualize_anomalies,\
    visualize_classification

class PeriodicMovingAveragePredictor(Predictor):
    def __init__(self, period, sigma):
        self.period = period
        self.sigma = sigma
        self.width = 2
    
    def fit(self):
        return

    def predict(self, data):
        values = data.value.values
        window = np.ones(self.width) / self.width
        values = np.convolve(values, window, 'same')
        
        min = self.period
        prev = 0
        for timestamp in data.timestamp.values:
            if timestamp - prev < min:
                min = timestamp - prev
            prev = timestamp
        
        bins = []
        for i in range(0, self.period, min):
            bins.append([])
        
        for i in range(len(data.timestamp.values)):
            timestamp = data.timestamp.values[i]
            value = values[i]
            index = int((timestamp % self.period) / min)
            bins[index].append(value)
        
        avgs = np.zeros(len(bins))
        stds = np.zeros(len(bins))
        for i in range(len(bins)):
            avgs[i] = np.mean(bins[i])
            stds[i] = np.std(bins[i])
        
        ret = pd.Series(np.zeros(len(data.timestamp.values), dtype=int))
        for i in range(len(values)):
            timestamp = data.timestamp.values[i]
            index = int((timestamp % self.period) / min)
            ret[i] = int(abs(avgs[index] - values[i]) > stds[index] * self.sigma)
        '''
        fig, ax = plt.subplots()
        ax.errorbar(range(0, self.period, min), avgs, yerr=stds)
        '''
        visualize_classification(data.timestamp.values, values, data.label.values, ret)
        
        return ret