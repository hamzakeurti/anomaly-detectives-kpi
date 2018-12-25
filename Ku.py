import pandas as pd
import numpy as np

import Util
import Analyze
import time
import matplotlib.pyplot as plt
import matplotlib

from extraction import Time, Seasonality
from extraction.Seasonality import preprocess
from predictors.MovingAveragePredictor import MovingAveragePredictor
from predictors.MovingAverageRollingStdPredictor import MovingAverageRollingStdPredictor
from predictors.RandomForestPredictor import RandomForestPredictor
from predictors.WeekOverWeekPredictor import WeekOverWeekPredictor
# from predictors.PeriodicPredictor import PeriodicPredictor

from Util import *
from extraction.Time import *
# matplotlib.use('Qt5Agg')
import pickle
from visualization.visualize import *
from predictors.PeriodicMovingAveragePredictor import PeriodicMovingAveragePredictor
from predictors.PeriodicDerivativePredictor import PeriodicDerivativePredictor
from predictors.PeriodicDerivativeMovingAveragePredictor import PeriodicDerivativeMovingAveragePredictor
from pprint import pprint, pformat

paramsPerKPI = {'02e99bd4f6cfb33f': {'sigma': 3, 'preprocess': False},
                '046ec29ddf80d62e': {'sigma': 3, 'preprocess': False},
                '07927a9a18fa19ae': {'sigma': 3, 'preprocess': True},
                '09513ae3e75778a3': {'sigma': 3.5, 'preprocess': False},
                '18fbb1d5a5dc099d': {'sigma': 3, 'preprocess': False},
                '1c35dbf57f55f5e4': {'sigma': 3, 'preprocess': False},
                '40e25005ff8992bd': {'sigma': 3, 'preprocess': False},
                '54e8a140f6237526': {'sigma': 3.5, 'preprocess': False},
                '71595dd7171f4540': {'sigma': 3.5, 'preprocess': False},  # try
                '769894baefea4e9e': {'sigma': 3.8, 'preprocess': False},
                '76f4550c43334374': {'sigma': 3.5, 'preprocess': False},
                '7c189dd36f048a6c': {'sigma': 3.5, 'preprocess': False},
                '88cf3a776ba00e7c': {'sigma': 3, 'preprocess': False},
                '8a20c229e9860d0c': {'sigma': 3.5, 'preprocess': False},
                '8bef9af9a922e0b3': {'sigma': 4, 'preprocess': False},
                '8c892e5525f3e491': {'sigma': 3, 'preprocess': False},
                '9bd90500bfd11edb': {'sigma': 5.5, 'preprocess': False},  # 3.5 already better
                '9ee5879409dccef9': {'sigma': 3, 'preprocess': False},
                'a40b1df87e3f1c87': {'sigma': 3.5, 'preprocess': False},  # GOOD
                'a5bf5d65261d859a': {'sigma': 4.5, 'preprocess': False},
                'affb01ca2b4f0b45': {'sigma': 3.5, 'preprocess': False},
                'b3b2e6d1a791d63a': {'sigma': 4, 'preprocess': False},
                'c58bfcbacb2822d1': {'sigma': 4, 'preprocess': False},
                'cff6d3c01e6a6bfa': {'sigma': 3, 'preprocess': False},
                'da403e4e3f87c9e0': {'sigma': 3, 'preprocess': False},
                'e0770391decc44ce': {'sigma': 3, 'preprocess': False}
                }

def main():

    logger = get_logger()
    unbeefed_train_data = load_train()
    split = split_on_id(unbeefed_train_data)
    ids_prediction = {}

    
    result_path = 'results/map_pdp'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    for id, df in split.items():
        og_df = df.copy()
        print(id)
        sigma = paramsPerKPI[id]['sigma']
        preprocess = paramsPerKPI[id]['preprocess']

        if preprocess:
            df = Seasonality.preprocess(df,refreshPickle=True)

        pdp = PeriodicDerivativePredictor(86400, 1, sigma)
        prediction = pdp.predict(df)

        if preprocess:
            prediction, _ = remove_imputed(prediction,df)
        ids_prediction[id] = prediction

        sections = Analyze.data_to_sections(og_df)
        adjusted_prediction = Analyze.adjust_prediction(prediction, sections, 7)
        precision, recall, fscore = Analyze.analyze(og_df, adjusted_prediction, 7)
        logger.info(f'{id}: precision = {precision:.3f}, recall = {recall:.3f}, f-score = {fscore:.3f}')
        # fig = visualize_classification(df.timestamp.values, df.value.values, df.label.values, adjusted_prediction)
        # plt.show()
        # fig.savefig(f'{result_path}/{id}.png')
    
    precision, recall, fscore = Analyze.analyze_per_id(split, ids_prediction, 7)
    logger.info(f'total: precision = {precision}, recall = {recall}, f-score = {fscore}')
    logger.info(pformat(paramsPerKPI))
if __name__ == '__main__':
    main()