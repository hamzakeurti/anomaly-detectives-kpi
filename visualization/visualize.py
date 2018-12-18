import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as clr

from Util import load_train
from extraction.Time import preprocess_train

KPI_ANOMALY_PLOTS_LOCATION = os.path.join('data','kpi_anomaly_plots')
KPI_ANOMALY_FIGS_LOCATION = os.path.join('data','kpi_anomaly_figs')
KPI_ID_LIST = ['02e99bd4f6cfb33f', '046ec29ddf80d62e', '07927a9a18fa19ae', '09513ae3e75778a3', '18fbb1d5a5dc099d', '1c35dbf57f55f5e4', '40e25005ff8992bd', '54e8a140f6237526', '71595dd7171f4540', '769894baefea4e9e', '76f4550c43334374', '7c189dd36f048a6c', '88cf3a776ba00e7c', '8a20c229e9860d0c', '8bef9af9a922e0b3', '8c892e5525f3e491', '9bd90500bfd11edb', '9ee5879409dccef9', 'a40b1df87e3f1c87', 'a5bf5d65261d859a', 'affb01ca2b4f0b45', 'b3b2e6d1a791d63a', 'c58bfcbacb2822d1', 'cff6d3c01e6a6bfa', 'da403e4e3f87c9e0', 'e0770391decc44ce']

def refresh_visualization_anomalies():
    '''
    Persists pyplots and png images of the KPI's with anomalies visualized
    '''
    if not os.path.exists(KPI_ANOMALY_PLOTS_LOCATION):
        os.makedirs(KPI_ANOMALY_PLOTS_LOCATION)
    if not os.path.exists(KPI_ANOMALY_FIGS_LOCATION):
        os.makedirs(KPI_ANOMALY_FIGS_LOCATION)
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
        plt.savefig(os.path.join(KPI_ANOMALY_FIGS_LOCATION,'imputed_kpi_%s' % kpi_id))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['blue', 'red']
        ax.scatter(no_impute_df.timestamp.values, no_impute_df.value.values,
                   marker='.',
                   c=no_impute_df.label.values, cmap=clr.ListedColormap(colors))
        ax.annotate(kpi_id, xy=(0.05, 0.95), xycoords='axes fraction')
        with open(os.path.join(KPI_ANOMALY_PLOTS_LOCATION, 'non_imputed_%s.pickle') % kpi_id, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(os.path.join(KPI_ANOMALY_FIGS_LOCATION,'non_imputed_kpi_%s' % kpi_id))
        plt.close()

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
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ['blue', 'red']
    ax.scatter(timestamp, values,
               marker='.',
               c=labels, cmap=clr.ListedColormap(colors))
    return fig

def visualize_classification(timestamp, values, labels, prediction):
    """plots KPI curve with highlight of TN TP FN FP
    """
    classification = 2 * prediction + labels
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ['blue', 'green', 'orange', 'red']
    ax.scatter(timestamp, values,
               marker='.',
               c=classification, cmap=clr.ListedColormap(colors))
    return fig
