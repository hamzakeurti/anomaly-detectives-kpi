import matplotlib.pyplot as plt
import matplotlib.colors as clr


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
