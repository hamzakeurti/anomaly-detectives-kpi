# %%
import argparse
from pprint import pformat

import pandas

import Util
import Analyze
import time
import matplotlib.pyplot as plt
import matplotlib
from PeriodicDerivativePredictor import PeriodicDerivativePredictor
import config
from extraction import Time, Seasonality
from predictors.MovingAveragePredictor import MovingAveragePredictor
from predictors.MovingAverageRollingStdPredictor import MovingAverageRollingStdPredictor
from predictors.RandomForestPredictor import RandomForestPredictor
from predictors.WeekOverWeekPredictor import WeekOverWeekPredictor
from Util import *
# matplotlib.use('Qt5Agg')
import pickle
from visualization.visualize import *

def main():


    args = getargs()
    split = split_on_id(load_train())

    if not os.path.isdir(args.outPath):
        os.makedirs(args.outPath)

    logger = get_logger()

    ids_prediction = {}
    for KPI_id, df in split.items():
        print('Predicting KPI: %s' % KPI_id)
        predictor = config.paramsPerKPI[args.config][KPI_id]['predictor']
        params = config.paramsPerKPI[args.config][KPI_id]['params']
        preprocess = config.paramsPerKPI[args.config][KPI_id]['preprocess']

        if preprocess:
            df = Seasonality.preprocess(df,refreshPickle=False)

        predictor = eval(predictor + '(' + params + ')')
        prediction = predictor.predict(df)
        if preprocess:
            prediction, df = Time.remove_imputed(prediction,df)
        ids_prediction[KPI_id] = prediction
        sections = Analyze.data_to_sections(df)
        adjusted_prediction = Analyze.adjust_prediction(prediction, sections, 7)
        precision, recall, fscore = Analyze.analyze(df, adjusted_prediction, 7)
        logger.info(f'{KPI_id}: precision = {precision:.3f}, recall = {recall:.3f}, f-score = {fscore:.3f}')

    precision, recall, fscore = Analyze.analyze_per_id(split, ids_prediction, 7)
    logger.info(f'total: precision = {precision}, recall = {recall}, f-score = {fscore}')
    logger.info(pformat(config.paramsPerKPI[args.config]))

    # # %%
    # # CHOOSE PREDICTOR
    # predictor = WeekOverWeekPredictor(long_term_width=60 * 24*7, season_widths=[60*24*7], sigma=3)
    # logger.info('Using predictor %s %s' % (str(predictor.__class__), str(predictor.__dict__)))
    # # Predict
    # start_time = time.time();
    # # only_one_kpi = {'9bd90500bfd11edb': beefed_data['9bd90500bfd11edb']}
    # #noise, season_components, long_term_component = hamzasExtractor(beefed_data)
    # #beefed_predictions = predictor.predict(noise)
    # beefed_predictions = predictor.predict(beefed_train_data)
    # unbeefed_predictions = Time.remove_imputed_predictions(beefed_predictions, beefed_train_data)
    # logger.info(f'Made predictions in {time.time() - start_time}s')

    # %%
    # Get anomalies using moving averages
    # raw_data_per_id = split_on_id(unbeefed_train_data)
    # for id in raw_data_per_id:
    #     # if id != '9bd90500bfd11edb':  # To focus on one id
    #     #     continue
    #     start_time = time.time()
    #
    #     # Visualize
    #     # visualize.visualize_classification(beefed_predictions   .index.values, noise.values,beefed_data[id].label.values,predicted.values)
    #     # Create blocks of anomalies from training data
    #     sections = Analyze.data_to_sections(raw_data_per_id[id])
    #     # Adjust predictions to blocks of anomalies
    #     adjusted_predictions = Analyze.adjust_prediction(unbeefed_predictions[id], sections, 7)
    #     # Draw graph
    #     #plt.figure(id)
    #     # Util.draw_graph(raw_data_per_id[id][:], adjusted_predictions[:])
    #     # precision, recall, fscore = Analyze.analyze_per_id({id: ids_data[id]}, {id: adjusted_predictions})
    #     precision, recall, fscore = Analyze.analyze(raw_data_per_id[id], adjusted_predictions)
    #     logger.info(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')
    #
    # #plt.show()


    #%% predict test data


def getargs():
    parser = argparse.ArgumentParser(description='Predict training data and save results to log')
    parser.add_argument('--out', dest='outPath', help='Where the output log will be stored'
                        , default=os.path.join('results', 'logs'))
    parser.add_argument('config', help='Name of the configuration in config.py to be used.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()