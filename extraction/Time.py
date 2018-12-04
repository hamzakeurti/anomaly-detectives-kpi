import datetime
import pandas as pd

def format_timestamp(dataframe):
    """Convert timestamp column to datetime object.
    Input dataframe should have timestamp column in int format."""
    dataframe.timestamp = dataframe.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x))


def extract_seasonal_time(dataframe):
    """Adds columns:
    minute (minutes of the day 0 to 60*24)
    and day (day of the week 0 to 6).
    Input dataframe should have timestamp column in datetime format,
    Use format_timestamp(dataframe)"""
    dataframe["minute"] = dataframe.timestamp.apply(lambda x: x.minute + 60*x.hour)
    dataframe["day"] = dataframe.timestamp.apply(lambda x: x.weekday())

def split_on_id(dataframe):
    """Returns a dictionary of dataframes for each id."""
    grouped_df = dataframe.groupby("KPI ID")
    return {x:grouped_df.get_group(x) for x in grouped_df.groups}


def fill_nas(single_KPI):
    """Pass a single KPI in, which timestamps column is already converted to datetime format"""
    #We need minutes column to proceed.
    extract_seasonal_time(single_KPI)


    single_KPI.index = pd.DatetimeIndex(single_KPI.timestamp)

    start = single_KPI.head(1).timestamp.iloc[0]
    end = single_KPI.tail(1).timestamp.iloc[0]
    gap = single_KPI.head(2).timestamp.iloc[1] - start

    indices = pd.date_range(start, end, freq=gap)

    single_KPI = single_KPI.reindex(indices)
    values_replace_na ={
        "KPI ID":single_KPI["KPI ID"].iloc[0],
        "label":0
    }
    single_KPI = single_KPI.fillna(values_replace_na)

    #Let's fill the value column now!
    #For each missing value at minute i of the day,
    #we will use the mean over all available values at same minute of different days
    #Note: we need times
    means = single_KPI.groupby(["minute"])["value"].describe()["mean"]
    single_KPI.timestamp = single_KPI.index
    na_KPI = single_KPI[single_KPI.minute != single_KPI.minute]
    new_values = na_KPI.timestamp.apply(lambda x:means[means.index == x.minute].values[0])
    single_KPI.value = single_KPI.value.fillna(new_values)
    extract_seasonal_time(single_KPI)
    return single_KPI

def preprocess(raw_dataframe):
    format_timestamp(raw_dataframe)
    ids_data = split_on_id(raw_dataframe)
    for KPI_id,df in ids_data.items():
        ids_data[KPI_id] = fill_nas(df)
    return ids_data