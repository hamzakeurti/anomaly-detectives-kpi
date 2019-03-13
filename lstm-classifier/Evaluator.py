from sklearn.metrics import precision_recall_fscore_support as score


class Evaluator:

    @staticmethod
    def bestFscoreForBinaryClassification(groundTruth, predictions):
        # Get the best fscore.
        bestFSCore, bestThreshold = 0, 0
        savedPredictedAnomolies = []

        for i in range(0, 100):
            predictedAnomolies = []
            threshold = i * 0.01
            for prediction in predictions:
                if (prediction > threshold):
                    predictedAnomolies.append(1)
                else:
                    predictedAnomolies.append(0)

            adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=groundTruth,
                                                                     predicted=predictedAnomolies)
            precision, recall, fscore, support = score(groundTruth, adjustedPredictedAnomolies)
            if (fscore[1] > bestFSCore):
                savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
                bestThreshold = threshold
                bestFSCore = fscore[1]

        return bestFSCore, bestThreshold, savedPredictedAnomolies

    @staticmethod
    def bestFscoreForScaledDiffPrediction(groundTruth, diffs):
        # Get the best fscore.
        bestFSCore, bestThreshold = 0, 0
        savedPredictedAnomolies = []

        for i in range(0, 100):
            predictedAnomolies = []
            threshold = i * 0.01
            for diff in diffs:
                if (diff > threshold):
                    predictedAnomolies.append(1)
                else:
                    predictedAnomolies.append(0)

            adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=groundTruth,
                                                                     predicted=predictedAnomolies)
            precision, recall, fscore, support = score(groundTruth, adjustedPredictedAnomolies)
            if (fscore[1] > bestFSCore):
                savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
                bestThreshold = threshold
                bestFSCore = fscore[1]

        return bestFSCore, bestThreshold, savedPredictedAnomolies

    @staticmethod
    def tpfptnfn(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)):
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1

        return(TP, FP, TN, FN)

    @staticmethod
    def data_to_sections(data):
        sections = []
        start = 0
        prev = 0
        for i in range(len(data)):
            curr = data[i]
            if curr:
                if not prev:
                    start = i
            else:
                if prev:
                    sections.append([start, i])
            prev = data[i]
        return sections
    @staticmethod
    def adjust_prediction(prediction, sections, T):
        adjusted_prediction = prediction.copy()
        for section in sections:
            is_true_positive = 0
            for i in range(section[0], min(section[0]+T+1, section[1])):
                if prediction[i]:
                    is_true_positive = 1
                    break
            for i in range(section[0], section[1]):
                adjusted_prediction[i] = is_true_positive
        return adjusted_prediction
    @staticmethod
    def getAdjustedFScore( groundTruth, predicted):
        sections = Evaluator.data_to_sections(groundTruth)
        adjustedPredictedData = Evaluator.adjust_prediction(predicted, sections, 7)
        return adjustedPredictedData
