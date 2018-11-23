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
