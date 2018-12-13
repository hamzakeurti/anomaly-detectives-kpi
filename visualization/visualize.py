import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as clr

from Util import load_train
from extraction.Time import preprocess_train

KPI_ANOMALY_PLOTS_LOCATION = os.path.join('data','kpi_anomaly_plots')

def refresh_visualization_anomalies():
    beefed_data = preprocess_train(load_train())
    kpis = beefed_data.keys()
    for kpi_id in kpis:
        df = beefed_data[kpi_id]
        no_impute_df = df[df['imputed'] == 0]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['blue', 'red']
        ax.scatter(df.timestamp.values, df.value.values,
                   marker='.',
                   c=df.label.values, cmap=clr.ListedColormap(colors))
        ax.annotate(kpi_id, xy=(0.05, 0.95), xycoords='axes fraction')
        with open(os.path.join(KPI_ANOMALY_PLOTS_LOCATION,'imputed_%s.pickle') % kpi_id, 'wb') as f:
            pickle.dump(fig, f)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['blue', 'red']
        ax.scatter(no_impute_df.timestamp.values, no_impute_df.value.values,
                   marker='.',
                   c=no_impute_df.label.values, cmap=clr.ListedColormap(colors))
        ax.annotate(kpi_id, xy=(0.05, 0.95), xycoords='axes fraction')
        with open(os.path.join(KPI_ANOMALY_PLOTS_LOCATION, 'non_imputed_%s.pickle') % kpi_id, 'wb') as f:
            pickle.dump(fig, f)
        break

def load_visualization_anomalies(kpi_id,imputed=False):
    '''
    Loads a pre-existing pyplot zoomable pic
    Imputed decides whether to include imputed samples or not
    '''
    if imputed:
        with open(os.path.join(KPI_ANOMALY_PLOTS_LOCATION,'imputed_%s.pickle') % kpi_id, 'rb') as f:
            return pickle.load(f)
    else:
        with open(os.path.join(KPI_ANOMALY_PLOTS_LOCATION,'non_imputed_%s.pickle') % kpi_id, 'rb') as f:
            return pickle.load(f)


#TODO make extra function that uses visualize_anomalies to quickly load figures for each KPI via pickling
def visualize_anomalies(timestamp, values, labels):
    """plots KPI with a highlight of anomalies
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['blue', 'red']
    ax.scatter(timestamp, values,
               marker='.',
               c=labels, cmap=clr.ListedColormap(colors))
    plt.show()


def visualize_classification(timestamp, values, labels, prediction):
    """plots KPI curve with highlight of TN TP FN FP
    """
    classification = 2 * prediction + labels
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['black', 'red', 'orange', 'green']
    ax.scatter(timestamp, values,
               marker='.',
               c=classification, cmap=clr.ListedColormap(colors))
    plt.show()
