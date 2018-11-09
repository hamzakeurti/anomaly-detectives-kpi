from numpy import exp, array, random, dot
import numpy as np
import pandas as pd
import trainer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



# IMPORTING DATA
Xy_train = pd.read_csv('./train.csv')
X_test = pd.read_csv('./test.csv')
X_train = Xy_train[['timestamp', 'value', 'KPI ID']]
y_train = Xy_train['label']

# WRITING OUTPUT
y_pred = trainer.predict(X_test) #Assuming a dataframe with columns 'KPI ID', timestamp, predict
y_pred.to_csv("./predict.csv")