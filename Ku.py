# %%
import pandas as pd
import numpy as np

import Util
import Analyze
import time
import matplotlib.pyplot as plt
import matplotlib

from extraction import Time
from predictors.MovingAveragePredictor import MovingAveragePredictor
from predictors.MovingAverageRollingStdPredictor import MovingAverageRollingStdPredictor
from predictors.RandomForestPredictor import RandomForestPredictor
from predictors.WeekOverWeekPredictor import WeekOverWeekPredictor
from predictors.PeriodicPredictor import PeriodicPredictor

from Util import *
from extraction.Time import *
# matplotlib.use('Qt5Agg')
import pickle
from visualization.visualize import *
from predictors.PeriodicMovingAveragePredictor import PeriodicMovingAveragePredictor
from predictors.PeriodicDerivativePredictor import PeriodicDerivativePredictor
from predictors.PeriodicDerivativeMovingAveragePredictor import PeriodicDerivativeMovingAveragePredictor

def main():
    logger = get_logger()
    start_time = time.time()
    unbeefed_train_data = load_train()

    split = split_on_id(unbeefed_train_data)
    ids_prediction = {}
    
    result_path = 'results/map_pdp'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    for id, df in split.items():
        map = MovingAveragePredictor(10, 10)
        #predictor = PeriodicPredictor(86400, 5)
        #predictor = PeriodicMovingAveragePredictor(86400, 3)
        pdp = PeriodicDerivativePredictor(86400, 1, 3)
        prediction = map.predict(df) | pdp.predict(df)
        
        ids_prediction[id] = prediction
        sections = Analyze.data_to_sections(df)
        adjusted_prediction = Analyze.adjust_prediction(prediction, sections, 7)

        precision, recall, fscore = Analyze.analyze(df, prediction, 7)
        logger.info(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')
        
        fig = visualize_classification(df.timestamp.values, df.value.values, df.label.values,
                                 adjusted_prediction)
        
        #plt.show()
        fig.savefig(f'{result_path}/{id}.png')
    
    precision, recall, fscore = Analyze.analyze_per_id(split, ids_prediction, 7)
    logger.info(f'total: precision = {precision}, recall = {recall}, f-score = {fscore}')


if __name__ == '__main__':
    main()