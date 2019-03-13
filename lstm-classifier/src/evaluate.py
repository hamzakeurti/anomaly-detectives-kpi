import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from Evaluator import Evaluator
from sklearn.metrics import precision_recall_fscore_support as score
from numpy import *
from Filler import Filler
from TrainingFormatter import TrainingFormatter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

kpi_id = "02e99bd4f6cfb33f"
inputDim = 100
modelToUse = "../savedModels/networkRun4/duringTraining/s1063"
df = pd.read_csv('../data/syntatic/02e99bd4f6cfb33ffilled.csv',sep='\t')
df = Filler.addNansToTheTimeseries(df, inputDim, kpi_id)
Filler.fillNansValues(df)

modelName = "leDeepNeuralNetwork" + kpi_id
timestampsSyntetic, valuesSyntetic, lablesSyntetic = df["timestamp"].values, df["value"].values, df["label"].values


scaler = StandardScaler()
valuesSyntetic, lables = scaler.fit_transform(valuesSyntetic.reshape(-1, 1)).flatten(), lablesSyntetic.flatten()
valuesPartitioned, valuesToPredict, timestamps = TrainingFormatter.predictOneTimestepAhed(inputDim, valuesSyntetic, lablesSyntetic, timestampsSyntetic)

#Split into X and Y's
test_portion = 0.3
test_n = int(len(valuesPartitioned) * test_portion)
xTrain, xTest = valuesPartitioned[:-test_n], valuesPartitioned[-test_n:]
yTrain, yTest = valuesToPredict[:-test_n], valuesToPredict[-test_n:]
tTrain, tTest = timestamps[:-test_n], timestamps[-test_n:]

model = load_model(modelToUse)

yPredict = model.predict(xTest)
yPredict = scaler.inverse_transform(yPredict)
predictedMap = {}
for i in range(0, len(tTest)):
    predictedMap[tTest[i]] = yPredict[i][0]


df = pd.read_csv('../data/train.csv')
df = df[df["KPI ID"] == kpi_id]
valuesReal, labelsReal, timestampsReal = df.value.values, df.label.values, df.timestamp.values
df["timestamp"] = df["timestamp"] - df.timestamp.values.min()

#Split into X and Y's
test_n = int(len(valuesReal) * test_portion)
_, xTestReal = valuesReal[:-test_n], valuesReal[-test_n:]
_, yTestReal = labelsReal[:-test_n], labelsReal[-test_n:]
_, tTestReal = timestampsReal[:-test_n], timestampsReal[-test_n:]

diffArray = []
newPredict = []
for i in range(0,len(tTestReal)):
    realValue, predictedValue = xTestReal[i], predictedMap[tTestReal[i]]
    diff = abs(predictedValue-realValue)
    newPredict.append(predictedValue)
    diffArray.append(diff)

# Get the best fscore.
bestFSCore = 0
savedPredictedAnomolies = []
for i in range(0,100):
    predictedAnomolies = []
    threshold = i*0.01
    for diff in diffArray:
        if (diff > threshold):
            predictedAnomolies.append(1)
        else:
            predictedAnomolies.append(0)

    adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=yTestReal,predicted=predictedAnomolies)
    precision, recall, fscore, support = score(yTestReal, predictedAnomolies)
    if (fscore[1] > bestFSCore):
        print("limit = ", threshold)
        savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
        bestFSCore = fscore[1]

print("beset fscore = {}".format(bestFSCore))
TP, FP, TN, FN = Evaluator.tpfptnfn(yTestReal,savedPredictedAnomolies)
print(TP, FP, TN, FN)


# plt.plot(diffArray, 'g') # plotting t, a separately
# plt.plot(yTest, 'r') # plotting t, b separately
plt.plot(savedPredictedAnomolies, 'b') # plotting t, c separately
plt.plot(newPredict, 'r')
plt.plot(xTestReal, 'g')
plt.show()