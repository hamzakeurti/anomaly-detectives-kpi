from predictors.RandomForestPredictor import RandomForestPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/train.csv')
dataFiltered = data[data["KPI ID"] == "02e99bd4f6cfb33f"]





xData = dataFiltered["value"].values.reshape(-1,1)
yData = pd.factorize(dataFiltered["label"])[0].reshape(-1,1)

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.2)


predicator = RandomForestPredictor()
predicator.train(xTrain, yTrain)
predictedValues = predicator.predict(xTest)


correctValues = 0
missedAnomolies = 0
for i in range(0, len(yTest)):
    if (yTest[i] == 1 and predictedValues[i] == 1):
        correctValues += 1
    else:
        missedAnomolies += 1

print("Num correct values = ", correctValues, " missed values = ", missedAnomolies)