import pandas as pd

baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
dfOuter = pd.read_csv(baseDir+'data/train.csv')

submission_df = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
for kpi_id in set(dfOuter["KPI ID"].values):
    df = pd.read_csv('results/'+kpi_id+'.csv')
    to_append = pd.DataFrame(columns=['KPI ID', 'timestamp', 'predict'])
    to_append['predict'], to_append['timestamp'], to_append['KPI ID'], = df.label.astype(
        int).values[1:], df.timestamp.values[1:], kpi_id
    submission_df = submission_df.append(to_append, ignore_index=True)


submission_df.to_csv("submission_5.csv", index=False)