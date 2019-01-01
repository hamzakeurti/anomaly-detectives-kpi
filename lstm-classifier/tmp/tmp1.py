# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Dropout
# from keras.models import load_model
# import keras
# import numpy as np
# from keras.layers import Dense
# from Evaluator import Evaluator
# from itertools import islice
# from sklearn.metrics import precision_recall_fscore_support as score
# from keras import backend as K
# from numpy import *
# import math
# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt
# from Filler import Filler
#
#
# kpi_id = "02e99bd4f6cfb33f"
# inputDim = 1440 # 1440
# df = pd.read_csv('../data/syntatic/02e99bd4f6cfb33ffilled.csv',sep='\t')
# df = Filler.addNansToTheTimeseries(df, inputDim, kpi_id)
# Filler.fillNansValues(df)
#
# modelName = "leDeepNeuralNetwork" + kpi_id
# timestampsSyntetic, valuesSyntetic, lablesSyntetic = df["timestamp"].values, df["value"].values, df["label"].values
#
# as_strided = np.lib.stride_tricks.as_strided
# scaler = StandardScaler()
# valuesSyntetic, lables = scaler.fit_transform(valuesSyntetic.reshape(-1, 1)).flatten(), lablesSyntetic.flatten()
# valuesPartitioned= as_strided(valuesSyntetic, (len(valuesSyntetic), inputDim), valuesSyntetic.strides * 2)[:-(inputDim)]
# valuesToPredict, timestamps = valuesSyntetic[(inputDim):], timestampsSyntetic[(inputDim):]
#
# #Split into X and Y's
# test_portion = 0.3
# test_n = int(len(valuesPartitioned) * test_portion)
# xTrain, xTest = valuesPartitioned[:-test_n], valuesPartitioned[-test_n:]
# yTrain, yTest = valuesToPredict[:-test_n], valuesToPredict[-test_n:]
# tTrain, tTest = timestamps[:-test_n], timestamps[-test_n:]
#
# print("xTest.len = {}, yTest.len = {}, tTest.len = {}".format(len(xTest), len(yTest), len(tTest)))
#
# saveModelCallback = keras.callbacks.ModelCheckpoint("/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/savedModels/networkRun4/duringTraining/s1", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#
#
# model = Sequential()
# model.add(Dense(units=inputDim, activation='relu', input_dim=inputDim, init='uniform'))
# model.add(Dropout(0.5))
# model.add(Dense(units=int((2 * inputDim) / 3), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=int((1 * inputDim) / 3), activation='relu'))
# model.add(Dense(units=1, activation='linear'))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(xTrain, yTrain, epochs=10, validation_data=(xTest, yTest), verbose=2, callbacks=[saveModelCallback])
# #
# # print("Model name = ", modelName)
# # # model = load_model("savedModels/networkRun1/"+ modelName)
# # # print("Model name = ", modelName)
# # # %%
# model.save("savedModels/networkRun3/" + modelName)
#
#
# # model = load_model("savedModels/networkRun3/" + modelName)
# yPredict = model.predict(xTest)
# # yPredict = yTest
# yPredict = scaler.inverse_transform(yPredict)
# predictedMap = {}
# for i in range(0, len(tTest)):
#     predictedMap[tTest[i]] = yPredict[i]
#
#
#
# df = pd.read_csv('../data/train.csv')
# df = df[df["KPI ID"] == kpi_id]
# valuesReal, labelsReal, timestampsReal = df.value.values, df.label.values, df.timestamp.values
# df["timestamp"] = df["timestamp"] - df.timestamp.values.min()
#
# #Split into X and Y's
# test_n = int(len(valuesReal) * test_portion)
# _, xTestReal = valuesReal[:-test_n], valuesReal[-test_n:]
# _, yTestReal = labelsReal[:-test_n], labelsReal[-test_n:]
# _, tTestReal = timestampsReal[:-test_n], timestampsReal[-test_n:]
#
# print("diffinLen = {}".format(len(tTest)-len(xTestReal)))
#
# diffArray = []
# # newPredict = []
# for i in range(0,len(tTestReal)):
#     realValue, predictedValue = xTestReal[i], predictedMap[tTestReal[i]]
#     diff = abs(predictedValue-realValue)
#     diffArray.append(diff)
#
# # Get the best fscore.
# bestFSCore = 0
# savedPredictedAnomolies = []
# # # brushStroke = 10
# for i in range(0,100):
#     predictedAnomolies = []
#     threshold = i*0.01
#     for diff in diffArray:
#         if (diff > threshold):
#             predictedAnomolies.append(1)
#         else:
#             predictedAnomolies.append(0)
#
#     adjustedPredictedAnomolies = Evaluator.getAdjustedFScore(groundTruth=yTestReal,predicted=predictedAnomolies)
#     precision, recall, fscore, support = score(yTestReal, predictedAnomolies)
#     if (fscore[1] > bestFSCore):
#         savedPredictedAnomolies = adjustedPredictedAnomolies.copy()
#         bestFSCore = fscore[1]
#
# print("beset fscore = {}".format(bestFSCore))
# TP, FP, TN, FN = Evaluator.tpfptnfn(yTest,savedPredictedAnomolies)
# print(TP, FP, TN, FN)
#
#
# # # plt.plot(diffArray, 'g') # plotting t, a separately
# # # plt.plot(yTest, 'r') # plotting t, b separately
# # # plt.plot(savedPredictedAnomolies, 'b') # plotting t, c separately
# # plt.plot(newPredict, 'r')
# # plt.plot(xTest, 'g')
 # plt.show()
import pandas as pd
from extraction import Time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

