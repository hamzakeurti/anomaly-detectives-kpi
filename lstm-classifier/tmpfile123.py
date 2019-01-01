import pickle
import pandas as pd
baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"

def getConfig2():
    with open(baseDir + 'results/lstmConfig.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        config = pickle.load(f)
    return config

df = pd.read_csv(baseDir + "data/test.csv")

config = getConfig2()
counter = 0
for id in set(df["KPI ID"].values):
    print("id = {}, config = {}".format(id, config[id]))
    counter += 1
    # print("{}".format(id) + config[id])
print(counter)
print(float(config["c58bfcbacb2822d1"]["threshold"][0]))