import numpy as np

def data_to_sections(data):
    sections = []
    start = 0
    prev = 0
    for i in range(len(data)):
        curr = data['label'][i]
        if curr:
            if not prev:
                start = i
        else:
            if prev:
                sections.append([start, i])
        prev = data['label'][i]
    return sections

def adjust_prediction(prediction, sections, T):
    adjusted_prediction = prediction.copy()
    for section in sections:
        is_true_positive = False
        for i in range(section[0], min(section[0]+T+1, section[1])):
            if prediction[i]:
                is_true_positive = True
                break
        for i in range(section[0], section[1]):
            adjusted_prediction[i] = is_true_positive
    return adjusted_prediction

def analyze(data, predictions, T=7):

    total_true_positive = 0
    total_selected = 0
    total_positive = 0
    sections = data_to_sections(data)
    adjusted_predictions = adjust_prediction(predictions, sections, T)
    n_true_positive = np.sum(adjusted_predictions & data['label'])
    n_selected = np.sum(adjusted_predictions)
    n_positive = np.sum(data['label'])
    total_true_positive  = total_true_positive + n_true_positive
    total_selected = total_selected + n_selected
    total_positive = total_positive + n_positive
    precision = 0
    if n_selected:
        precision = n_true_positive / n_selected
    recall = 0
    if n_positive:
        recall = n_true_positive / n_positive
    fscore = 0
    if precision + recall:
        fscore = 2 * precision * recall / (precision + recall)

    return precision, recall, fscore


def analyze_per_id(ids_data, ids_predictions, T=7):
    sections = {}
    adjusted_predictions = {}
    total_true_positive = 0
    total_selected = 0
    total_positive = 0
    for id in ids_data:
        data = ids_data[id]
        predictions = ids_predictions[id]
        sections = data_to_sections(data)
        adjusted_predictions = adjust_prediction(predictions, sections, T)
        n_true_positive = np.sum(adjusted_predictions & data['label'])
        n_selected = np.sum(adjusted_predictions)
        n_positive = np.sum(data['label'])
        total_true_positive  = total_true_positive + n_true_positive
        total_selected = total_selected + n_selected
        total_positive = total_positive + n_positive
    
    precision = 0
    if n_selected:
        precision = n_true_positive / n_selected
    recall = 0
    if n_positive:
        recall = n_true_positive / n_positive
    fscore = 0
    if precision+recall:
        fscore = 2 * precision * recall / (precision + recall)
        
    return precision, recall, fscore