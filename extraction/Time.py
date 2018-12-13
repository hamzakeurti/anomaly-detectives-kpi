import datetime
import os
import pickle

import pandas as pd

pd.options.mode.chained_assignment = None  # THis to ignore the SettingWithCopyWarning (https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas)

TRAIN_BEEFED_PICKLE_PATH = "data/train_beefed.p"
TEST_BEEFED_PICKLE_PATH = "data/test_beefed.p"


def format_timestamp(dataframe):
    """Convert timestamp column to datetime object.
    Input dataframe should have timestamp column in int format."""
    dataframe.timestamp = dataframe.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x))


def extract_seasonal_time(dataframe):
    """Adds columns:
    minute (minutes of the day 0 to 60*24-1)
    and day (day of the week 0 to 6).
    Input dataframe should have timestamp column in datetime format,
    Use format_timestamp(dataframe)"""
    dataframe["minute"] = get_minute_of_day_column(dataframe.timestamp)
    dataframe["day"] = get_day_of_week_column(dataframe.timestamp)
    dataframe["minute_of_week"] = dataframe.minute + 60 * 24 * dataframe.day


# TODO probs move this out of Time file, into util
def split_on_id(dataframe):
    """Returns a dictionary of dataframes for each id."""
    grouped_df = dataframe.groupby("KPI ID")
    return {x: grouped_df.get_group(x) for x in grouped_df.groups}


def get_minute_of_day_column(timestamps):
    return timestamps.apply(lambda x: x.minute + 60 * x.hour)


def get_day_of_week_column(timestamps):
    return timestamps.apply(lambda x: int(x.weekday()))


def fill_nas(single_KPI, not_anomaly=False):
    """Pass a single KPI in, which timestamps column is already converted to datetime format"""
    # We need minutes column to proceed.
    extract_seasonal_time(single_KPI)

    single_KPI.index = pd.DatetimeIndex(single_KPI.timestamp)

    start = single_KPI.head(1).timestamp.iloc[0]
    end = single_KPI.tail(1).timestamp.iloc[0]
    gap = single_KPI.head(2).timestamp.iloc[1] - start

    indices = pd.date_range(start, end, freq=gap)

    single_KPI["imputed"] = 0  # Whether or not the row is imputed
    single_KPI = single_KPI.reindex(indices)
    values_replace_na = {
        "KPI ID": single_KPI["KPI ID"].iloc[0],
        "label": 0,
        "imputed": 1
    }
    single_KPI = single_KPI.fillna(values_replace_na)

    # Let's fill the value column now!
    # For each missing value at minute i of the week,
    # we will use the mean over all available values at same minute of different weeks
    # Note: we need times

    if not_anomaly:  # We can choose to only consider non anomalous values in the computing of means.
        means = single_KPI[single_KPI.label == 0].groupby(["minute"])["value"].mean()
    else:  # We cannot select non anomalous during testing. (TODO: In case of testing fill with means from training)
        means = single_KPI.groupby(["minute"])["value"].mean()

    # assert len(means)*gap.minute + gap.hour*60 == 60*24 # Check if we have a value for each minute of the day
    single_KPI.timestamp = single_KPI.index
    na_KPI = single_KPI[single_KPI.imputed == 1]
    extract_seasonal_time(na_KPI)

    new_values = na_KPI["minute"].apply(lambda x: means[x])
    values_replace_na = {
        "value": new_values,
        "minute": na_KPI.minute.values,
        "day": na_KPI.day.values,
        "minute_of_week": na_KPI.minute_of_week
    }
    single_KPI.value = single_KPI.value.fillna(new_values)
    return single_KPI


def preprocess_train(raw_dataframe, train_beefed_pickle_path=TRAIN_BEEFED_PICKLE_PATH, refreshPickle=False):
    if (not os.path.exists(TRAIN_BEEFED_PICKLE_PATH)) or refreshPickle:
        format_timestamp(raw_dataframe)
        beefed_data = split_on_id(raw_dataframe)
        for KPI_id, df in beefed_data.items():
            print(KPI_id)
            beefed_data[KPI_id] = fill_nas(df,not_anomaly=True)
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
            beefed_data[KPI_id] = fill_nas(df, not_anomaly=False)
        pickle.dump(beefed_data, open(test_beefed_pickle_path, "wb"))
    else:
        beefed_data = pickle.load(open(test_beefed_pickle_path, "rb"))
    return beefed_data


def remove_imputed_predictions(ids_predictions, beefed_data):
    for id in ids_predictions:
        pred = ids_predictions[id]
        imputed = beefed_data[id].imputed
        both = list(zip(pred, imputed))
        unbeefed = [pr for (pr, im) in both if not im]
        ids_predictions[id] = pd.Series(unbeefed)
    return ids_predictions
