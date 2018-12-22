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
import argparse


def predict_per_kpi(test_df, args):
    submission_df = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
    split = split_on_id(test_df)
    for id, df in split.items():
        print('Predicting KPI: %s' % id)
        predictor = eval(args.predictor)(*args.params)
        prediction = predictor.predict(df)
        to_append = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
        to_append['predict'], to_append['timestamp'], to_append['KPI ID'], = prediction.astype(
            int).values, df.timestamp.values, id
        submission_df = submission_df.append(to_append, ignore_index=True)
    return submission_df


def predict(test_df, args):
    submission_df = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
    predictor = eval(args.predictor)(*args.params)
    prediction = predictor.predict(test_df)
    submission_df['predict'], submission_df['timestamp'], submission_df['KPI ID'], = prediction.astype(
        int).values, test_df.timestamp.values, test_df.id.values


def getargs():
    parser = argparse.ArgumentParser(description = 'Make submission file with given predictor')
    parser.add_argument('predictor',
                        help='name of the predictor to use')
    parser.add_argument('--params', dest='params', nargs='+')
    parser.add_argument('--out', dest='outPath', default=os.path.join('results','submissions'))
    parser.add_argument('--per_kpi', dest='outPath', default='results')
    parser.add_argument('--not_per_kpi', dest='not_per_kpi', action='store_true',default=False)
    args = parser.parse_args()
    args.params = Util.extra_parse(args.params)
    return args


def main():
    args = getargs()
    test_df = load_test()

    if not os.path.isdir(args.outPath):
        os.makedirs(args.outPath)

    if args.not_per_kpi:
        submission_df = predict(test_df,args)
    else:
        submission_df = predict_per_kpi(test_df,args)
    submission_df.to_csv(os.path.join(args.outPath,'submission_%s.csv' % time.time()), index=False)


if __name__ == '__main__':
    main()