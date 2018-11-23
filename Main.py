#%%
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
#matplotlib.use('Qt5Agg')


FILE_NAME = 'data/train.csv'#'train/54e8a140f6237526.csv'
# Read file
start_time = time.time()
#raw_data = pandas.read_csv(FILE_NAME)
ids_data = Util.file_name_to_ids_datas(FILE_NAME)
print(f'Read file {FILE_NAME} in {time.time() - start_time}s')
#%%
# CHOOSE PREDICTOR
predictor = WeekOverWeekPredictor(long_term_width=60*24,season_widths=[60*24],sigma=3)
# Predict
start_time = time.time();
ids_predictions = predictor.predict(ids_data)
print(f'Made predictions in {time.time() - start_time}s')

#%%
# Get anomalies using moving averages
for id in ids_data:
    # if id != '046ec29ddf80d62e': #To focus on one id
    #     continue
    start_time = time.time()
    # Create blocks of anomalies from training data
    sections = Analyze.data_to_sections(ids_data[id])
    # Adjust predictions to blocks of anomalies
    adjusted_predictions = Analyze.adjust_prediction(ids_predictions[id], sections, 7)
    # Draw graph
    plt.figure(id)
    Util.draw_graph(ids_data[id][:], adjusted_predictions[:])
    #precision, recall, fscore = Analyze.analyze_per_id({id: ids_data[id]}, {id: adjusted_predictions})
    precision, recall, fscore = Analyze.analyze(ids_data[id], adjusted_predictions)
    print(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')

plt.show()