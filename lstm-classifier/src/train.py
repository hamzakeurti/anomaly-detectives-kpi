#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import load_model
import keras
import numpy as np
from keras.layers import Dense
from Evaluator import Evaluator
from itertools import islice
from sklearn.metrics import precision_recall_fscore_support as score
from keras import backend as K
from numpy import *
import math
import matplotlib
from sklearn.externals import joblib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from Filler import Filler

baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
kpi_id = "e0770391decc44ce"
inputDim = 10
n_batch = 1
dfOuter = pd.read_csv(baseDir+'data/train.csv')

for kpi_id in set(dfOuter["KPI ID"].values):
    kpi_id = "e0770391decc44ce"
    df = pd.read_csv(baseDir+'data/syntatic/'+kpi_id+'filled.csv', sep='\t')
    df["timestamp"] = df["timestamp"] - df.timestamp.values.min()
    df = Filler.addNansToTheTimeseries(df, inputDim, kpi_id)
    Filler.fillNansValues(df)

    modelName = "leDeepNeuralNetwork" + kpi_id
    timestampsSyntetic, valuesSyntetic, lablesSyntetic = df["timestamp"].values, df["value"].values, df["label"].values

    as_strided = np.lib.stride_tricks.as_strided
    scaler = MinMaxScaler(feature_range=(-1,1))

    valuesSyntetic, lables = scaler.fit_transform(valuesSyntetic.reshape(-1, 1)).flatten(), lablesSyntetic.flatten()
    print("saved")
    joblib.dump(scaler, baseDir + "savedModels/networkRun8/scaler")
    valuesPartitioned= as_strided(valuesSyntetic, (len(valuesSyntetic), inputDim), valuesSyntetic.strides * 2)[:-(inputDim)]
    valuesToPredict, timestamps = valuesSyntetic[(inputDim):], timestampsSyntetic[(inputDim):]

    #Split into X and Y's
    test_portion = 0.3
    lower_portion = 0.3
    test_n = int(len(valuesPartitioned) * test_portion)
    xTrain, xTest = valuesPartitioned[:-test_n], valuesPartitioned[-test_n:]
    yTrain, yTest = valuesToPredict[:-test_n], valuesToPredict[-test_n:]
    tTrain, tTest = timestamps[:-test_n], timestamps[-test_n:]

    print("xTest.len = {}, yTest.len = {}, tTest.len = {}".format(len(xTest), len(yTest), len(tTest)))

    saveModelCallback = keras.callbacks.ModelCheckpoint(baseDir+"savedModels/networkRun8/duringTraining/s1",
                                                        verbose=0,
                                                        save_weights_only=False, mode='auto', period=1)

    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
    xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

    model = Sequential()
    model.add(LSTM(6, input_shape=(xTrain.shape[1], xTrain.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(xTrain, yTrain, epochs=10, batch_size=25, verbose=1, shuffle=False)

    model.save(baseDir + "savedModels/networkRun8/" + modelName )
    # load_model(baseDir + "savedModels/networkRun8/" + modelName)
    valuesPartitioned = np.reshape(valuesPartitioned, (valuesPartitioned.shape[0], 1, valuesPartitioned.shape[1]))
    yPredict = model.predict(valuesPartitioned)

    yPredict = scaler.inverse_transform(yPredict)
    predictedMap = {}

    for i in range(0, len(timestamps)):
        predictedMap[timestamps[i]] = yPredict[i]

    df = pd.read_csv( baseDir + 'data/train.csv')
    df = df[df["KPI ID"] == kpi_id]
    valuesReal, labelsReal, timestampsReal = df.value.values, df.label.values, df.timestamp.values
    df["timestamp"] = df["timestamp"] - df.timestamp.values.min()

    #Split into X and Y's
    test_n = int(len(valuesReal) * test_portion)
    _, xTestReal = valuesReal[:-test_n], valuesReal[-test_n:]
    _, yTestReal = labelsReal[:-test_n], labelsReal[-test_n:]
    _, tTestReal = timestampsReal[:-test_n], timestampsReal[-test_n:]

    print("diffinLen = {}".format(len(tTest)-len(xTestReal)))

    newPredict = []
    diffArray = []
    for i in range(0,len(tTestReal)):
        realValue, predictedValue = xTestReal[i], predictedMap[tTestReal[i]]
        diff = abs(predictedValue-realValue)
        diffArray.append(diff)
        newPredict.append(predictedMap[tTestReal[i]])

    # Get the best fscore.
    bestFSCore, bestThreshold = 0, 0
    savedPredictedAnomolies = []
    # loopLimit = int(np.max(diffArray))
    diffScaler = MinMaxScaler(feature_range=(0,1))
    diffArray = diffScaler.fit_transform(diffArray)
    for i in range(0,100):
        predictedAnomolies = []
        threshold = i*0.01
        for diff in diffArray:
            if (diff > threshold):
                predictedAnomolies.append(1)
            else:
                predictedAnomolies.append(0)

        adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=yTestReal,predicted=predictedAnomolies)
        precision, recall, fscore, support = score(yTestReal, adjustedPredictedAnomolies)
        if (fscore[1] > bestFSCore):
            savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
            TP, FP, TN, FN = Evaluator.tpfptnfn(yTestReal, predictedAnomolies)
            bestThreshold = threshold
            bestFSCore = fscore[1]
    print(bestThreshold)
    # print(type(bestThreshold))
    # bestThresholdRescaled = diffScaler.inverse_transform(np.array([bestThreshold]).reshape(1,-1))
    fileToWriteScore = open("../savedModels/networkRun6/result",'a')
    printString = "KPI ID = {} beset fscore = {}\n"\
        .format(kpi_id, bestFSCore)
    fileToWriteScore.write(printString)
    # print(printString)
    break

# plt.plot(yPredict, 'r')
# plt.plot(valuesReal, 'g')
# plt.plot(diffArray, 'black')
# plt.plot((labelsReal), 'b')
# plt.show()