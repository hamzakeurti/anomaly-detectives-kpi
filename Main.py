import Util
import Analyze
import time
import matplotlib.pyplot as plt
from MovingAveragePredictor import MovingAveragePredictor
from MovingAverageRollingStdPredictor import MovingAverageRollingStdPredictor

FILE_NAME = 'train.csv'#'train/54e8a140f6237526.csv'

# Read file
start_time = time.time()
ids_data = Util.file_name_to_ids_datas(FILE_NAME)
print(f'Read file {FILE_NAME} in {time.time() - start_time}s')


# Predict
map = MovingAveragePredictor(50, 5)
start_time = time.time();
ids_predictions = map.predict(ids_data)
print(f'Made predictions in {time.time() - start_time}s')

# Get anomalies using moving averages
for id in ids_data:
    start_time = time.time()
    # Create blocks of anomalies from training data
    sections = Analyze.data_to_sections(ids_data[id])
    # Adjust predictions to blocks of anomalies
    adjusted_predictions = Analyze.adjust_prediction(ids_predictions[id], sections, 7)
    # Draw graph
    plt.figure(id)
    Util.draw_graph(ids_data[id][:], adjusted_predictions[:])
    precision, recall, fscore = Analyze.analyze({id: ids_data[id]}, {id: adjusted_predictions})
    print(f'{id}: precision = {precision}, recall = {recall}, f-score = {fscore}')
    break

plt.show()