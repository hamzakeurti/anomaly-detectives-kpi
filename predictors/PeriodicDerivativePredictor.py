import datetime

from extraction import Seasonality
from extraction.Time import format_timestamp, unformat_timestamp, remove_imputed
from predictors.PredictorTemplate import Predictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization.visualize import visualize_anomalies, \
    visualize_classification





class PeriodicDerivativePredictor(Predictor):

    def __init__(self, period, width, sigma):
        self.period = period
        self.width = width
        self.sigma = sigma

    def fit(self):
        return

    def predict(self, data):

        kpi = data["KPI ID"].iloc[0]
        # If preprocessed, this will be changed
        if isinstance(data['timestamp'].iloc[0], datetime.datetime):
            unformat_timestamp(data)

        values = data.value.values
        values = values[self.width:] - values[:-self.width]
        timestamps = data.timestamp.values
        dtimestamps = timestamps[self.width:] - timestamps[:-self.width]
        values = values / dtimestamps
        timestamps = timestamps[self.width:]

        min = self.period
        prev = 0
        for timestamp in timestamps:
            if timestamp - prev < min:
                min = timestamp - prev
            prev = timestamp

        bins = []
        for i in range(0, self.period, min):
            bins.append([])

        for i in range(len(timestamps)):
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
            ret[self.width + i] = int(abs(avgs[index] - values[i]) > stds[index] * self.sigma)  # self.sigma)
        '''
        fig, ax = plt.subplots()
        ax.errorbar(range(0, self.period, min), avgs, yerr=stds)
        '''
        return ret
