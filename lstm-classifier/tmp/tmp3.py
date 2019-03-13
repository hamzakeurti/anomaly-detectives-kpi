from extraction import Time
import pandas as pd


baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
kpi_id = "e0770391decc44ce"

inputDim = 1
nb_epoch = 10

# Setup
modelToUsePath = baseDir + "savedModels/testRun1/trainNetwork1" + kpi_id + ".mdl"
scalerToUsePath = baseDir + "savedModels/testRun1/scalerNetwork1" + kpi_id

df = pd.read_csv("../data/test.csv")
values, timestamps = df["value"].values, df["timestamp"].values

print(df.shape)
Time.format_timestamp(df)
df["label"] = 0
print(df)
# df = Time.fill_nas(df,True)
print(df.shape)