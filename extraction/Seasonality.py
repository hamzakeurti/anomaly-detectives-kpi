import os

import pandas as pd
from extraction import Time

MINUTE_FROM_START = "minute_from_start"
BIG_TREND_MEANS = "big_trend_means"
BIG_TREND_STDS = "big_trend_stds"
BIG_TREND_EXTRACTED = "big_trend_extracted"
WEEKLY_EXTRACTED = "weekly_extracted"
EXTRACTED_DAILY = "extracted_daily"
PREPPED_TRAIN_PICKLE_FOLDER = os.path.join('data','prepped_train')

def preprocess(single_KPI, pickle_folder=PREPPED_TRAIN_PICKLE_FOLDER, refreshPickle=False,ignore_anomaly=True):
    if not os.path.exists(pickle_folder):
        os.makedirs(pickle_folder)
    kpi = single_KPI['KPI ID'].iloc[0]
    pickle_path = os.path.join(pickle_folder,kpi + '.p')
    if (not os.path.exists(pickle_path)) or refreshPickle:

        Time.format_timestamp(single_KPI)
        single_KPI = Time.fill_nas(single_KPI,ignore_anomaly)
        # single_KPI = extend_timeseries(single_KPI)
        extract_big_trend(single_KPI)
        extract_weekly_seasonality(single_KPI)
        extract_daily_seasonality(single_KPI)

        #This to deal with the extended values. TODO Might want to mark those as extended, and give proper values for all columns
        # single_KPI['KPI ID'] = kpi
        # single_KPI['imputed'] = single_KPI['imputed'].map({float('NaN'): 1, 0: 0})
        # Replace value columns by the extracted daily, for later use
        to_return = single_KPI.loc[:, single_KPI.columns != 'value'].rename(columns={'extracted_daily': 'value'})
        to_return.to_pickle(pickle_path)
    else:
        to_return = pd.read_pickle(pickle_path)
    return to_return


def extend_timeseries(single_KPI, to_extend="value", timedelta="7 days"):
    """

    :param single_KPI: Must have gone through Time.fillna()
    :param timedelta:
    :return:
    """
    # We use the means to fill the extended df
    means = single_KPI.groupby('minute')[to_extend].mean()

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
    extended_values = extended_df[extended_df[to_extend].isna()]['minute'].apply(lambda x: means[x])
    extended_df = extended_df.fillna({to_extend: extended_values, 'label': 0})

    return extended_df


def extract_big_trend(single_KPI, window_width_minutes=1440 * 7):
    """
    :param dataframe: evenly spaced series, use Time.preprocess first otherwise
    :param window_width:
    :return: adds a column to the dataframe for the rolling trend
    """
    start = single_KPI.timestamp[0]
    end = single_KPI.timestamp[-1]

    extended_df = extend_timeseries(single_KPI)
    big_trend_means = extended_df.value.rolling('7D').mean()
    big_trend_stds = extended_df.value.rolling('7D').std()
    # The rolling places the value at the right edge,
    # let's adjust it by translating the obtained values 3.5days to the left.
    big_trend_means.index = big_trend_means.index - pd.Timedelta('3.5D')
    big_trend_stds.index = big_trend_stds.index - pd.Timedelta('3.5D')

    single_KPI[BIG_TREND_MEANS] = big_trend_means[start:end]
    single_KPI[BIG_TREND_STDS] = big_trend_stds[start:end]
    single_KPI[BIG_TREND_EXTRACTED] = (single_KPI['value'] - single_KPI[BIG_TREND_MEANS]) / single_KPI[BIG_TREND_STDS]


def extract_weekly_seasonality(single_KPI):
    start = single_KPI.timestamp[0]
    end = single_KPI.timestamp[-1]

    extended_extracted = extend_timeseries(single_KPI=single_KPI, to_extend=BIG_TREND_EXTRACTED)
    daily_average = extended_extracted[BIG_TREND_EXTRACTED].rolling('1D').mean()
    daily_average.index = daily_average.index - pd.Timedelta('0.5D')

    single_KPI["daily_averages"] = daily_average[start:end]
    mean_week = single_KPI.groupby('minute_of_week')['daily_averages'].mean()
    single_KPI[WEEKLY_EXTRACTED] = single_KPI.apply(lambda x: x[BIG_TREND_EXTRACTED] - mean_week[x['minute_of_week']],
                                                    axis=1)


def extract_daily_seasonality(single_KPI):
    means = single_KPI.groupby('minute')[WEEKLY_EXTRACTED].mean()

    single_KPI[EXTRACTED_DAILY] = single_KPI.apply(lambda x: x[WEEKLY_EXTRACTED] - means[x['minute']], axis=1)

# def extract_seasonal_component(single_KPI,big_width,small_width):
#     '''
#     Adds a column to single_KPI that contains repetitions with period big_width of moving averages width window size
#     small_width applied to the value column.
#     For big trend: big_width = the entire period of single_KPI, there is only one repitition, and small_width would typically be week.
#     For eg weekly trend: big_widht = one week, this will repeat as many times as there are weeks in the entire period, and small_width could be day
#     :param single_KPI: Evenly spaced series *that can contain NaN values*
#     '''
#     season_averages = get_averages_per_period #TODO This is nice and general, but maybe hard to let play well with pandas timedelta, groupby, ...

# def extract_big_trend_ignore_imputed(single_KPI):

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
