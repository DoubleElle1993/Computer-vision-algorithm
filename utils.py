import pandas as pd
import config
from datetime import datetime


def get_season(date):
    """
    This function enables to distinguish the analysis by season temperatures.
    """


    if isinstance(date, datetime):
        date = date.date()
    date = date.replace(year=config.Y)
    return next(season for season, (start, end) in config.seasons
                if start <= date <= end)


def remove_key(dictionary, key1, key2):
    """
    This function allows to delete some keys from a dictionary.
    """


    d_copy = dict(dictionary)
    del d_copy[key1][key2]

    return d_copy


def create_csv_output(dictionary):
    """
    This function provides a csv output with json keys as columns.
    """


    try:
        df = pd.read_csv('Output/output.csv', index_col=0)
    except:
        row = pd.json_normalize(dictionary)
        row.columns = config.csv_output_columns
        row = row[config.readable_columns_csv]
        df = pd.DataFrame(columns=row.columns)
        df.to_csv('Output/output.csv')
        df = pd.read_csv('Output/output.csv', index_col=0)
    row = pd.json_normalize(dictionary)
    row.columns = config.csv_output_columns
    row = row[config.readable_columns_csv]
    df = df.append(row, ignore_index=True)
    df.to_csv('Output/output.csv')
    print('The Csv file is saved')

    return df
