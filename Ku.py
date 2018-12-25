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
from config import *

def main():

    logger = get_logger()
    train_df = load_train()
    split = split_on_id(train_df)
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