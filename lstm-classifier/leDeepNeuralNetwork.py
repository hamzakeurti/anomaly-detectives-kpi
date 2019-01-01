import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout
from keras.models import load_model
import numpy as np
from keras.layers import Dense
from Evaluator import Evaluator
from itertools import islice
from sklearn.metrics import precision_recall_fscore_support as score
from keras import backend as K
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def create_subseq(data, lookBackLength):
    sub_seq = []
    for i in range(0, len(data) - lookBackLength):
        sub_seq.append(data[i:i+lookBackLength])
    return sub_seq

as_strided = np.lib.stride_tricks.as_strided
kpi_id = "02e99bd4f6cfb33f"
inputDim = 10
df = pd.read_csv('data/train.csv')

for kpi_id in set(df["KPI ID"].values):
    try:
        kpi_id = "76f4550c43334374"
        data = df[df["KPI ID"] == kpi_id]
        modelName = "leDeepNeuralNetworkInitial" + kpi_id

        timestamps, values, lables = data["timestamp"].values, data["value"].values, data["label"].values

        # values = StandardScaler().fit_transform(values.reshape(-1, 1)).flatten()
        lables = lables.flatten()

        valuesCompressed = as_strided(values, (len(values), inputDim), values.strides * 2)[:-(inputDim-1)]
        lablesCompressed = lables[(inputDim - 1):]
        #
        xTrain, xTest, yTrain, yTest = train_test_split(valuesCompressed, lablesCompressed, test_size=0.3,
                                                        shuffle=False)

        #
        model = Sequential()
        model.add(Dense(units=inputDim, activation='relu', input_dim=inputDim, init='uniform'))
        model.add(Dropout(0.5))
        model.add(Dense(units=int((2 * inputDim) / 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=50, validation_data=(xTest, yTest), verbose=2)

        print("Model name = ", modelName)
        # model = load_model("savedModels/networkRun1/"+ modelName)
        # print("Model name = ", modelName)
        # %%
        yPredict = model.predict(xTest)
        savedFscore = 0
        # for i in range(0, 100):
        # predictSaveFile = open("results/pureResults/" + modelName + ".txt", "w")
        # for predict in yPredict:
        # predictSaveFile.write(str(predict) + "\n")
        # predictSaveFile.close()
        formatedPredictions = []
        threshold = float(0.9)
        for value in yPredict:
            if (value[0] > threshold):
                formatedPredictions.append(1)
            else:
                formatedPredictions.append(0)
        plt.plot(yPredict, color = 'b')
        plt.plot(yTest, color='g')
        plt.plot(xTest, color='r')
        plt.show()


        # precision, recall, fscore, support = score(y_true=yTest, y_pred=formatedPredictions)
        # print("Total fscore = {}".format(fscore))
        # numberOfZero = 0
        # for predicted in formatedPredictions:
        #     if (predicted == 1):
        #         numberOfZero += 1
        # print("predictedZero = {}".format(numberOfZero))
        #
        # formatedPredictions = []
        # threshold = float(0.9)
        # for value in yPredict:
        #     if (value[0] > threshold):
        #         formatedPredictions.append(1)
        #     else:
        #         formatedPredictions.append(0)
        #
        # precision, recall, fscore, support = Evaluator.getAdjustedFScore(groundTruth=yTest,
        #                                                                  predicted=formatedPredictions)
        # model.save("savedModels/networkRun1/" + modelName)
        # print(fscore)
        # if (len(fscore) > 1):
        #
        #     if (fscore[1] > savedFscore):
        #         print(fscore)
        #         savedFscore = fscore[1]
        #
        #     resultLine = "Kpi: {}, fscore: {}, precision: {}, recall: {}, support: {} \n".format(kpi_id, fscore,
        #                                                                                          precision,
        #                                                                                          recall, support)
        #     print(resultLine)
        #     fileToWrite = open("results/networkRun1", "a")
        #     fileToWrite.write(resultLine)
        #     fileToWrite.close()
        break
            # except Exception as  e:
            #     print("exception ", e)
    except Exception as e:
        print(e)
    K.clear_session()
