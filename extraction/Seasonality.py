import pandas as pd


MINUTE_FROM_START = "minute_from_start"

def extract_big_trend(single_KPI,window_width_minutes=1440*7):
    """
    :param dataframe: evenly spaced series, use Time.preprocess first otherwise
    :param window_width:
    :return: adds a column to the dataframe for the rolling trend
    :return: adds a column to the dataframe for the values minus the trend
    """
    period = single_KPI.minute[1] - single_KPI.minute[0]
    window_size = window_width_minutes / period
    #TODO:We might want to complete the dataframe to the left and to the right with an average week.

    single_KPI["big_trend"] = \
        single_KPI.values.rolling_mean(window=window_size)
    single_KPI["trend_extracted"] = single_KPI.values - single_KPI["big_trend"]


def add_minutes_from_start(single_KPI):
    gap = int(single_KPI.head(2).timestamp.iloc[1] - single_KPI.head(2).timestamp.iloc[0])
    single_KPI[MINUTE_FROM_START] = gap * single_KPI.index

def extract_seasonal(single_KPI,season_minutes,window_width_minutes=None,from_column="values",not_anomaly=False):
    period = single_KPI.minute[1] - single_KPI.minute[0]
    if not window_width_minutes:
        window_width_minutes = period

    if not_anomaly:  # We can choose to only consider non anomalous values in the computing of means.
        means = single_KPI[single_KPI.label == 0].groupby([groupby_column])["value"].mean()
    else:  # We cannot select non anomalous during testing. (TODO: In case of testing fill with means from training)
        means = single_KPI.groupby([groupby_column])["value"].mean()
