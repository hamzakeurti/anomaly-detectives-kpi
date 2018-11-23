from predictors.PredictorTemplate import Predictor
import pandas as pd
import numpy as np
from operator import add
import matplotlib.pyplot as plt
import operator

#TODO Maybe use median rather than average for the moving average
#TODO make season width expressible in minutes rather than index
class WeekOverWeekPredictor(Predictor):
    def __init__(self, long_term_width, season_widths, sigma):
        """

        :param long_term_width: Window size for the long-term trend moving average, ideally longer than any seasonality, expressed in minutes
        :param season_widths: List of window sizes for the seasonal moving averages, one for each suspected seasonality,
        """
        self.long_term_width = long_term_width
        self.season_widths = season_widths
        self.sigma = sigma

    def fit(self):
        return

    def predict(self, datas):
        predictions = {}
        i = 0
        for id in datas:

            # if id != '046ec29ddf80d62e':  # To focus on one id
            #     continue
            data = datas[id]
            values = data['value'].values
            timestamps = data['timestamp'].values
            period_in_minutes = (timestamps[1]-timestamps[0])/60
            print(period_in_minutes)
            # labels = data['label']
            long_term_width_in_index = int(self.long_term_width//period_in_minutes)
            seasonal_widths_in_index = list(map(int,self.season_widths//period_in_minutes))

            to_convolve = np.ones(long_term_width_in_index) / long_term_width_in_index
            long_term_component = np.convolve(values, to_convolve, 'same')

            no_long_term_component = np.subtract(values,long_term_component)



            noise = self.get_no_seasonal_components(no_long_term_component, seasonal_widths_in_index)
            noise = pd.Series(noise)
            plt.plot(values, label='values')
            plt.plot(long_term_component, label='long_term_component')
            plt.plot(no_long_term_component, label='no_long_term_component')
            plt.plot(noise, label='noise')
            plt.legend()
            plt.show()

            std = np.std(noise)
            predictions[id] = pd.Series(abs(noise) > std * self.sigma)
        return predictions
    def get_no_seasonal_components(self, no_long_term_component, seasonal_widths_in_index):
        seasonal_components = np.zeros(len(no_long_term_component))
        most_deseasoned_so_far = no_long_term_component

        #To be correct, remove trends from big to small
        for sw in sorted(seasonal_widths_in_index,reverse=True):
            nb_seasons = len(no_long_term_component) // sw
            season_averages = np.zeros(sw)
            for i in range(sw):
                same_time_in_season_points = most_deseasoned_so_far[i::sw]
                season_averages[i] = sum(same_time_in_season_points)/len(same_time_in_season_points)
            # Expand season averages to length of values
            season_averages = np.tile(season_averages,nb_seasons+1)[:len(no_long_term_component)]

            plt.plot(season_averages, label='season_averages for season width %s' % sw)
            most_deseasoned_so_far = list(map(operator.sub, most_deseasoned_so_far, season_averages))
        return most_deseasoned_so_far


# w = WeekOverWeekPredictor(11,[6,6],1.96) #TODO make the test some data series with two different frequencies
# d = {'id1':pd.DataFrame({'value':[1,5,12,3,7,14,5,9,16,7,11,18,9,13,20],'timestamp':list(range(0,1500,100)),'label':np.ones(15)})}
# plt.plot(d.get('id1')['value'].values)
# p = w.predict(d)
# print(p)