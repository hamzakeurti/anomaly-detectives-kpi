import pandas as pd
from extraction import Time

MINUTE_FROM_START = "minute_from_start"


def extend_timeseries(single_KPI, timedelta="7 days"):
    """

    :param single_KPI: Must have gone through Time.fillna()
    :param timedelta:
    :return:
    """
    # We use the means to fill the extended df
    means = single_KPI.groupby('minute').value.mean()

    start = single_KPI.timestamp[0]
    end = single_KPI.timestamp[-1]
    new_start = start - pd.Timedelta(timedelta)
    new_end = end + pd.Timedelta(timedelta)
    gap = single_KPI.timestamp[1] - start

    extended_df = single_KPI.copy()
    new_indices = pd.date_range(new_start, new_end, freq=gap)
    extended_df = extended_df.reindex(new_indices)
    extended_df.timestamp = extended_df.index

    # Now let's fill the minute columns, we'll need it to fill the extended values.
    Time.extract_seasonal_time(extended_df)
    extended_values = extended_df[extended_df.value.isna()]['minute'].apply(lambda x: means[x])
    extended_df = extended_df.fillna({'value':extended_values,'label':0})

    return extended_df

# def extract_big_trend(single_KPI,window_width_minutes=1440*7):
#     """
#     :param dataframe: evenly spaced series, use Time.preprocess first otherwise
#     :param window_width:
#     :return: adds a column to the dataframe for the rolling trend
#     :return: adds a column to the dataframe for the values minus the trend
#     """
#     period = single_KPI.minute[1] - single_KPI.minute[0]
#     window_size = window_width_minutes / period
#     #TODO:We might want to complete the dataframe to the left and to the right with an average week.
#
#     single_KPI["big_trend"] = \
#         single_KPI.value.rolling_mean(window=window_size)
#     single_KPI["trend_extracted"] = single_KPI.values - single_KPI["big_trend"]


# def add_minutes_from_start(single_KPI):
#     gap = int(single_KPI.head(2).timestamp.iloc[1] - single_KPI.head(2).timestamp.iloc[0])
#     single_KPI[MINUTE_FROM_START] = gap * single_KPI.index

# def extract_seasonal(single_KPI,season_minutes,window_width_minutes=None,from_column="values",not_anomaly=False):
#     period = single_KPI.minute[1] - single_KPI.minute[0]
#     if not window_width_minutes:
#         window_width_minutes = period
#
#     if not_anomaly:  # We can choose to only consider non anomalous values in the computing of means.
#         means = single_KPI[single_KPI.label == 0].groupby([groupby_column])["value"].mean()
#     else:  # We cannot select non anomalous during testing. (TODO: In case of testing fill with means from training)
#         means = single_KPI.groupby([groupby_column])["value"].mean()
