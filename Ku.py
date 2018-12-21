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
# from predictors.PeriodicPredictor import PeriodicPredictor

from Util import *
from extraction.Time import *
# matplotlib.use('Qt5Agg')
import pickle
from visualization.visualize import *
from predictors.PeriodicMovingAveragePredictor import PeriodicMovingAveragePredictor
from predictors.PeriodicDerivativePredictor import PeriodicDerivativePredictor
from predictors.PeriodicDerivativeMovingAveragePredictor import PeriodicDerivativeMovingAveragePredictor

def main():
    submission_df = pd.DataFrame(columns=['KPI ID','timestamp','predict'])

    logger = get_logger()
    start_time = time.time()
    unbeefed_test_data = load_test()
    # unbeefed_test_data = load_train()
    split = split_on_id(unbeefed_test_data)
    ids_prediction = {}

    
    result_path = 'results/map_pdp'
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    df_sum, ta_sum = 0,0
    for id, df in split.items():
        print(id)
        # map = MovingAveragePredictor(10, 10)
        #predictor = PeriodicPredictor(86400, 5)
        #predictor = PeriodicMovingAveragePredictor(86400, 3)
        pdp = PeriodicDerivativePredictor(86400, 1, 3)
        prediction = pdp.predict(df) #map.predict(df) |
        
        ids_prediction[id] = prediction
        # sections = Analyze.data_to_sections(df)
        # adjusted_prediction = Analyze.adjust_prediction(prediction, sections, 7)
        to_append = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
        to_append['predict'], to_append['timestamp'] , to_append['KPI ID'],  = prediction.astype(int).values, df.timestamp.values, id #pd.DataFrame({'KPI ID':df['KPI ID'], 'timestamp':df.timestamp, 'predict':prediction.astype(int)})
        submission_df = submission_df.append(to_append,ignore_index=True)
        print("Df shape: ",df.shape)
        print("Shape",to_append.shape)
        df_sum += df.shape[0]
        ta_sum += to_append.shape[0]
        print("DF sum so far: ", df_sum)
        print("ta sum so far: ", ta_sum)
        # precision, recall, fscore = Analyze.analyze(df, prediction, 7)
        # logger.info(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')
        #
        # fig = visualize_classification(df.timestamp.values, df.value.values, df.label.values,
        #                          adjusted_prediction)

        #plt.show()
        # fig.savefig(f'{result_path}/{id}.png')
    
    # precision, recall, fscore = Analyze.analyze_per_id(split, ids_prediction, 7)
    # logger.info(f'total: precision = {precision}, recall = {recall}, f-score = {fscore}')
    print(submission_df.shape)
    submission_df.to_csv('results/submission_%s.csv' % time.time(),index=False)

if __name__ == '__main__':
    main()