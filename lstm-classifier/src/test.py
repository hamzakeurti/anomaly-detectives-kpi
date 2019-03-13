#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from Evaluator import Evaluator
from keras.models import load_model
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support as score
from TrainingFormatter import TrainingFormatter
from numpy import *
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from extraction import Time
from Filler import Filler
import pickle
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support as score
# encoding=utf8

def getPreprossesedDataframe(kpi_id, nansToAdd):
    df = pd.read_csv(baseDir + 'data/train.csv')
    df = df[df["KPI ID"] == kpi_id]
    # df["label"] = 0
    return df

def shiftValuesAndLabels(df,numberOfNansToAdd, kpi):
    df = Filler.addNansToTheTimeseries(df, numberOfNansToAdd, kpi)
    Filler.fillNansValues(df)
    return df

def loadConfig():
    # global baseDir
    # with open(baseDir + '/results/lstmConfig.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    #     config = pickle.load(f)

    config = {}
    fileLines = open(baseDir + "data/config").readlines()
    for line in fileLines:
        parts = line.rstrip("\n").split("=")
        config[parts[0]] = parts[1]

    return config

def getModelFromKpi(kpi, index):
    path = baseDir + "savedModels/testRun{}/trainNetwork1{}.mdl".format(index, kpi)
    return load_model(path)

def getScalerForKpi(kpi, index):
    path = baseDir + "savedModels/testRun{}/scalerNetwork{}{}".format(index, index, kpi)
    return joblib.load(path)

baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
kpi_id = "e0770391decc44ce"
config = loadConfig()
inputDim = 10

# Setup
dfOuter = pd.read_csv(baseDir + 'data/train.csv')
submission_df = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])


predicted = []
groundTrouths = []
# ---Prediction begin---
for kpi_id in set(dfOuter["KPI ID"].values):
    privousBestFScore = 0
    # for iteration in range(0,10):

    df = getPreprossesedDataframe(kpi_id, inputDim)
    df = shiftValuesAndLabels(df,inputDim, kpi_id)
    # for iteration in range(0,int(config[kpi_id]["iteration"])):
    values, timestamps, labels = df["value"].values, df["timestamp"].values, df["label"].values[inputDim:]
    i=1
    model = getModelFromKpi(kpi_id, i+1)
    scaler = getScalerForKpi(kpi_id, i+1)

    scaledValues = scaler.transform(values.reshape(-1, 1)).flatten()
    valuesPartitioned = np.lib.stride_tricks.as_strided(scaledValues, (len(scaledValues), inputDim), scaledValues.strides * 2)[:-(inputDim)]
    valuesPartitioned = valuesPartitioned.reshape(valuesPartitioned.shape[0], valuesPartitioned.shape[1], 1)

    predictedData = scaler.inverse_transform(model.predict(valuesPartitioned))
    groundTrouth = scaler.inverse_transform(scaledValues.reshape(-1,1))[inputDim:]

    diffScaler = MinMaxScaler(feature_range=(0,1))
    diff = diffScaler.fit_transform(abs(predictedData-groundTrouth))

    for k,d in enumerate(diff):
        if (d.item() >= float(config[kpi_id])):
            # df["predict"][k+inputDim] = 1
            df["value"][k+inputDim] = predictedData[k][0]

    bestFScore, treshold, adjustedPredictedAnomalies = Evaluator.bestFscoreForScaledDiffPrediction(labels, diff)
    absThresh = diffScaler.inverse_transform(np.reshape(treshold, (1,-1)))
    print("bestFSCore = {}".format(bestFScore))
    for value in adjustedPredictedAnomalies:
        predicted.append(value)
    for value in labels:
        groundTrouths.append(value)
precision, recall, fscore, support = score(groundTrouths, predicted)
print("fscore = " , fscore)