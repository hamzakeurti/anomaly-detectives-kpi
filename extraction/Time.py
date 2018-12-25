import datetime
import os
import pickle

import pandas as pd

pd.options.mode.chained_assignment = None  # THis to ignore the SettingWithCopyWarning (https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)

TRAIN_BEEFED_PICKLE_PATH = "data/train_beefed.p"
TEST_BEEFED_PICKLE_PATH = "data/test_beefed.p"

# Column names
KPI_ID = "KPI ID"
VALUE = "value"
MINUTE = "minute"
DAY = "day"
MINUTE_OF_WEEK = "minute_of_week"
LABEL = "label"
IMPUTED = "imputed"


def format_timestamp(dataframe):
    """
    Modifies input
    Convert timestamp column to datetime object.
    Input dataframe should have timestamp column in int format.
    """
    dataframe.timestamp = dataframe.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x))


def unformat_timestamp(df):
    """
    Modifies input
    Converts datetime object to int indicating its seconds
    """
    df.timestamp = df.timestamp.apply(lambda x: int(datetime.datetime.timestamp(x)))


def extract_seasonal_time(dataframe):
    """Modifies input. Adds columns:
    minute (minutes of the day 0 to 60*24-1)
    and day (day of the week 0 to 6).
    Input dataframe should have timestamp column in datetime format,
    Use format_timestamp(dataframe)"""
    dataframe[MINUTE] = get_minute_of_day_column(dataframe.timestamp)
    dataframe[DAY] = get_day_of_week_column(dataframe.timestamp)
    dataframe[MINUTE_OF_WEEK] = dataframe.minute + 60 * 24 * dataframe.day


# TODO probs move this out of Time file, into util
def split_on_id(dataframe):
    """Returns a dictionary of dataframes for each id."""
    grouped_df = dataframe.groupby(KPI_ID)
    return {x: grouped_df.get_group(x) for x in grouped_df.groups}


def get_minute_of_day_column(timestamps):
    return timestamps.apply(lambda x: x.minute + 60 * x.hour)


def get_day_of_week_column(timestamps):
    return timestamps.apply(lambda x: int(x.weekday()))


def fill_nas_NaN_value(single_KPI):
    '''
    :param single_KPI: Dataframe with single KPI and unformatted timestamp
    :return:
    '''
    extract_seasonal_time(single_KPI)

    single_KPI.index = pd.DatetimeIndex(single_KPI.timestamp)

    start = single_KPI.head(1).timestamp.iloc[0]
    end = single_KPI.tail(1).timestamp.iloc[0]
    gap = single_KPI.head(2).timestamp.iloc[1] - start

    indices = pd.date_range(start, end, freq=gap)

    single_KPI[IMPUTED] = 0  # Whether or not the row is imputed
    single_KPI = single_KPI.reindex(indices)
    values_replace_na = {
        KPI_ID: single_KPI[KPI_ID].iloc[0],
        LABEL: 0,
        IMPUTED: 1,
        VALUE: float('NaN')
    }
    single_KPI = single_KPI.fillna(values_replace_na)
    single_KPI.timestamp = single_KPI.index
    return single_KPI


def fill_nas(single_KPI, ignore_anomaly=False):
    """
    :returns filled input.
    Pass a single KPI in, which timestamps column is already converted to datetime format
    """
    # We need minutes column to proceed.
    new_single_KPI = single_KPI.copy()

    extract_seasonal_time(new_single_KPI)

    new_single_KPI.index = pd.DatetimeIndex(new_single_KPI.timestamp)

    start = new_single_KPI.head(1).timestamp.iloc[0]
    end = new_single_KPI.tail(1).timestamp.iloc[0]
    gap = new_single_KPI.head(2).timestamp.iloc[1] - start

    indices = pd.date_range(start, end, freq=gap)

    new_single_KPI[IMPUTED] = 0  # Whether or not the row is imputed
    new_single_KPI = new_single_KPI.reindex(indices)
    values_replace_na = {
        KPI_ID: new_single_KPI[KPI_ID].iloc[0],
        LABEL: 0,
        IMPUTED: 1
    }
    new_single_KPI = new_single_KPI.fillna(values_replace_na)

    # Let's fill the value column now!
    # For each missing value at minute i of the week,
    # we will use the mean over all available values at same minute of different weeks
    # Note: we need times

    if ignore_anomaly:  # We cannot select non anomalous during testing. (TODO: In case of testing fill with means from training)
        means = new_single_KPI.groupby([MINUTE])[VALUE].mean()
    else:  # We can choose to only consider non anomalous values in the computing of means.
        means = new_single_KPI[new_single_KPI.label == 0].groupby([MINUTE])[VALUE].mean()

    # assert len(means)*gap.minute + gap.hour*60 == 60*24 # Check if we have a value for each minute of the day
    new_single_KPI.timestamp = new_single_KPI.index
    na_KPI = new_single_KPI[new_single_KPI.value.isna()]
    if len(na_KPI) != 0:  # extract_seasonal_time breaks otherwise
        extract_seasonal_time(na_KPI)

        new_values = na_KPI[MINUTE].apply(lambda x: means[x])
        values_replace_na = {
            VALUE: new_values,
            MINUTE: na_KPI.minute,
            DAY: na_KPI.day,
            MINUTE_OF_WEEK: na_KPI.minute_of_week
        }
        new_single_KPI = new_single_KPI.fillna(values_replace_na)
        # single_KPI = single_KPI.reset_index(drop=True)
    return new_single_KPI


def preprocess_train(raw_dataframe, train_beefed_pickle_path=TRAIN_BEEFED_PICKLE_PATH, refreshPickle=False):
    if (not os.path.exists(train_beefed_pickle_path)) or refreshPickle:
        print("fgdsgdge", train_beefed_pickle_path)
        format_timestamp(raw_dataframe)
        beefed_data = split_on_id(raw_dataframe)
        for KPI_id, df in beefed_data.items():
            print(KPI_id)
            beefed_data[KPI_id] = fill_nas(df, ignore_anomaly=False)
        pickle.dump(beefed_data, open(train_beefed_pickle_path, "wb"))
    else:
        beefed_data = pickle.load(open(train_beefed_pickle_path, "rb"))
    return beefed_data


def preprocess_test(raw_dataframe, test_beefed_pickle_path=TEST_BEEFED_PICKLE_PATH, refreshPickle=False):
    if (not os.path.exists(TEST_BEEFED_PICKLE_PATH)) or refreshPickle:
        format_timestamp(raw_dataframe)
        beefed_data = split_on_id(raw_dataframe)
        for KPI_id, df in beefed_data.items():
            print(KPI_id)
            beefed_data[KPI_id] = fill_nas(df, ignore_anomaly=True)
        pickle.dump(beefed_data, open(test_beefed_pickle_path, "wb"))
    else:
        beefed_data = pickle.load(open(test_beefed_pickle_path, "rb"))
    return beefed_data


def remove_imputed(predicted, input):
    """
    Removes rows from predicted for which the value in the column 'imputed' in input is 1.
    Assumes predicted and input are dataframes whose rows correspond
    """
    input['predicted'] = list(predicted)
    return input.loc[input['imputed'] == 0]['predicted']


def remove_imputed_predictions(ids_predictions, beefed_data):
    for id in ids_predictions:
        pred = ids_predictions[id]
        imputed = beefed_data[id].imputed
        both = list(zip(pred, imputed))
        unbeefed = [pr for (pr, im) in both if not im]
        ids_predictions[id] = pd.Series(unbeefed)
    return ids_predictions
