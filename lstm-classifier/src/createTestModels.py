#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from Evaluator import Evaluator
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
from numpy import *
from Filler import Filler
from sklearn.externals import joblib

baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
kpi_id = "e0770391decc44ce"
modelName = "trainNetwork1"
inputDim = 10
nb_epoch = 10
dfOuter = pd.read_csv(baseDir+'data/train.csv')
runIndex = 2

for kpi_id in set(dfOuter["KPI ID"].values):
    print("Kpi_id = {}".format(kpi_id))
    modelNameWithKpiId = "trainNetwork1" + kpi_id

    df = pd.read_csv(baseDir+'data/syntatic/'+kpi_id+'filled.csv', sep='\t')
    df["timestamp"] = df["timestamp"] - df.timestamp.values.min()
    df = Filler.addNansToTheTimeseries(df, inputDim, kpi_id)
    Filler.fillNansValues(df)

    timestampsSyntetic, valuesSyntetic, lablesSyntetic = df["timestamp"].values, df["value"].values, df["label"].values

    as_strided = np.lib.stride_tricks.as_strided
    scaler = MinMaxScaler(feature_range=(-1,1))
    valuesSyntetic, lables = scaler.fit_transform(valuesSyntetic.reshape(-1, 1)).flatten(), lablesSyntetic.flatten()
    valuesPartitioned = as_strided(valuesSyntetic, (len(valuesSyntetic), inputDim), valuesSyntetic.strides * 2)[:-(inputDim)]
    valuesToPredict, timestamps = valuesSyntetic[(inputDim):], timestampsSyntetic[(inputDim):]

    joblib.dump(scaler, baseDir + "savedModels/testRun{}/scalerNetwork{}".format(str(runIndex),str(runIndex)+kpi_id) )

    xTrain = valuesPartitioned
    yTrain = valuesToPredict

    xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    model = Sequential()
    model.add(LSTM(6, input_shape=(xTrain.shape[1], xTrain.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    trainingScore = model.fit(xTrain, yTrain, epochs=10, batch_size=25, verbose=2, shuffle=False)
    model.save(baseDir+"savedModels/testRun{}/{}.mdl".format(runIndex, modelNameWithKpiId))

    valuesPartitioned = np.reshape(valuesPartitioned, (valuesPartitioned.shape[0], valuesPartitioned.shape[1], 1))
    yPredict = model.predict(valuesPartitioned)
    yPredict = scaler.inverse_transform(yPredict)
    predictedMap = {}
    for i in range(0, len(timestamps)):
        predictedMap[timestamps[i]] = yPredict[i]

    df = pd.read_csv( baseDir + 'data/train.csv')
    df = df[df["KPI ID"] == kpi_id]
    valuesReal, labelsReal, timestampsReal = df.value.values, df.label.values, df.timestamp.values
    df["timestamp"] = df["timestamp"] - df.timestamp.values.min()

    diffArray = []
    for i in range(0,len(valuesReal)):
        realValue, predictedValue = valuesReal[i], predictedMap[timestampsReal[i]]
        diff = abs(predictedValue-realValue)
        diffArray.append(diff)

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

        adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=labelsReal,predicted=predictedAnomolies)
        precision, recall, fscore, support = score(labelsReal, adjustedPredictedAnomolies)
        if (fscore[1] > bestFSCore):
            savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
            bestThreshold = threshold
            bestFSCore = fscore[1]
    fileToWriteScore = open(baseDir+"savedModels/testRun{}/traiedModelResult".format(runIndex),'a')
    printString = "KPI ID = {} beset fscore = {}, train_loss: {}, threshold = {}\n"\
        .format(kpi_id, bestFSCore, trainingScore.history["loss"][-1], bestThreshold)
    fileToWriteScore.write(printString)
    print(printString)
    fileToWriteScore.close()
