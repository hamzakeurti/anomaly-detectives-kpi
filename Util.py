import logging
import pickle
import time

import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import tz

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TRAIN_RAW_PICKLE_PATH = "data/train_raw.p"
TEST_PICKLE_PATH = "data/test.p"
LOG_PATH = "results/logs"


# Separate file content by ids
def file_content_to_ids_data(file_content):
    unique_ids = pd.unique(file_content['KPI ID'])
    ids_data = {}
    for unique_id in unique_ids:
        data = file_content[file_content['KPI ID'] == unique_id]
        data = data.reset_index(drop=True)
        ids_data[unique_id] = data
    return ids_data


def file_name_to_ids_datas(file_name):
    file_content = pd.read_csv(file_name)
    ids_datas = file_content_to_ids_data(file_content)
    return ids_datas


# Save into different files
def data_to_file(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(f'{path}/{id}.csv', index=False)


def get_results(ids_data, ids_predictions):
    ids_results = ids_data.copy()
    for id in ids_results:
        if 'label' in ids_results[id]:
            ids_results[id] = ids_results[id].drop(columns='label')
        if 'value' in ids_results[id]:
            ids_results[id] = ids_results[id].drop(columns='value')
        ids_results[id]['predict'] = ids_predictions[id].astype(int)
    return pd.concat([ids_results[id] for id in ids_results])


# Draw graph
def draw_graph(data, predictions):
    tp = data[data['label'] & predictions]  # True positive
    fp = data[~data['label'] & predictions]  # False positive
    tn = data[~data['label'] & ~predictions]  # True negative
    fn = data[data['label'] & ~predictions]  # False negative
    plt.plot(tn['timestamp'], tn['value'], color='#000000',
             marker='.', linestyle='')
    plt.plot(tp['timestamp'], tp['value'], color='tab:red',
             marker='.', linestyle='')
    plt.plot(fp['timestamp'], fp['value'], color='tab:orange',
             marker='.', linestyle='')
    plt.plot(fn['timestamp'], fn['value'], color='tab:green',
             marker='.', linestyle='')


# Convert timestamp to datetime
def timestamp_to_datetime(timestamp):
    utc = datetime.utcfromtimestamp(timestamp).replace(tzinfo=tz.gettz('UTC'))
    china = utc.astimezone(tz.gettz('Asia/Shanghai'))
    return china


def load_train(train_path=TRAIN_PATH, train_raw_pickle_path=TRAIN_RAW_PICKLE_PATH, refreshPickle=False):
    if (not os.path.exists(train_raw_pickle_path)) or refreshPickle:
        raw_data = pd.read_csv(train_path)
        pickle.dump(raw_data, open(train_raw_pickle_path, "wb"))
    else:
        raw_data = pickle.load(open(train_raw_pickle_path, "rb"))
    return raw_data


def load_test(test_path=TEST_PATH, test_pickle_path=TEST_PICKLE_PATH, refreshPickle=False):
    if (not os.path.exists(test_pickle_path)) or refreshPickle:
        data = pd.read_csv(test_path)
        pickle.dump(data, open(test_pickle_path, "wb"))
    else:
        data = pickle.load(open(test_pickle_path, "rb"))
    return data

def get_logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(LOG_PATH)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger