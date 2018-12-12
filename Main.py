# %%
import pandas

import Util
import Analyze
import time
import matplotlib.pyplot as plt
import matplotlib
from predictors.MovingAveragePredictor import MovingAveragePredictor
from predictors.MovingAverageRollingStdPredictor import MovingAverageRollingStdPredictor
from predictors.RandomForestPredictor import RandomForestPredictor
from predictors.WeekOverWeekPredictor import WeekOverWeekPredictor
from Util import *
from extraction import Time
# matplotlib.use('Qt5Agg')
import pickle
from visualization import visualize

def main():
    # TODO make predictor return unbeefed
    # TODO change preprocessed to including imputed column. Find imputable rows for that
    # FILE_NAME = 'data/train.csv'  # 'train/54e8a140f6237526.csv'
    # Read file
    # raw_data = pandas.read_csv(FILE_NAME)
    # ids_data = Util.file_name_to_ids_datas(FILE_NAME)
    #
    # #TODO move this into load_train/test
    # if not os.path.exists('data/raw_data.p'):
    #     raw_data = load_train()
    #     print(f'Read file {FILE_NAME} in {time.time() - start_time}s')
    #     pickle.dump(raw_data, open("data/raw_data.p", "wb"))
    # else:
    #     raw_data = pickle.load(open("data/raw_data.p", "rb"))
    #     print(f'Loaded file {FILE_NAME} in {time.time() - start_time}s')
    # small_data = raw_data.head(100)
    #
    # if not os.path.exists('data/beefed_data.p'):
    #     beefed_data = Time.preprocess(small_data)
    #     pickle.dump(beefed_data, open("data/beefed_data.p", "wb"))
    # else:
    #     beefed_data = pickle.load(open("data/beefed_data.p", "rb"))

    logger = get_logger()
    start_time = time.time()
    unbeefed_train_data = load_train()
    unbeefed_test_data = load_test()
    logger.info('Loaded data in % s seconds' % (time.time() - start_time))
    start_time = time.time()
    beefed_train_data = Time.preprocess(unbeefed_train_data)
    beefed_test_data = Time.preprocess(unbeefed_test_data)
    logger.info('Loaded preprocessed data in % s seconds' % (time.time() - start_time))

    # %%
    # CHOOSE PREDICTOR
    predictor = WeekOverWeekPredictor(long_term_width=60 * 24*7, season_widths=[60*24*7], sigma=3)
    # Predict
    start_time = time.time();
    # only_one_kpi = {'9bd90500bfd11edb': beefed_data['9bd90500bfd11edb']}
    #noise, season_components, long_term_component = hamzasExtractor(beefed_data)
    #beefed_predictions = predictor.predict(noise)
    beefed_predictions = predictor.predict(beefed_train_data)
    unbeefed_predictions = Time.remove_imputed_samples(beefed_predictions, beefed_train_data)
    print(f'Made predictions in {time.time() - start_time}s')

    # %%
    # Get anomalies using moving averages
    raw_data_per_id = file_content_to_ids_data(unbeefed_train_data)
    for id in raw_data_per_id:
        # if id != '9bd90500bfd11edb':  # To focus on one id
        #     continue
        start_time = time.time()

        # Visualize
        # visualize.visualize_classification(beefed_predictions   .index.values, noise.values,beefed_data[id].label.values,predicted.values)
        # Create blocks of anomalies from training data
        sections = Analyze.data_to_sections(raw_data_per_id[id])
        # Adjust predictions to blocks of anomalies
        adjusted_predictions = Analyze.adjust_prediction(unbeefed_predictions[id], sections, 7)
        # Draw graph
        #plt.figure(id)
        # Util.draw_graph(raw_data_per_id[id][:], adjusted_predictions[:])
        # precision, recall, fscore = Analyze.analyze_per_id({id: ids_data[id]}, {id: adjusted_predictions})
        precision, recall, fscore = Analyze.analyze(raw_data_per_id[id], adjusted_predictions)
        print(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')

    #plt.show()


    #%% predict test data

    beefed_test_predictions = predictor.predict(beefed_test_data)
    unbeefed_predictions = Time.remove_imputed_samples(beefed_test_predictions, beefed_test_data)


if __name__ == '__main__':
    main()