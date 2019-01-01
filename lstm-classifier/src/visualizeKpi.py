import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as clr




kpi_id = "02e99bd4f6cfb33f"
df = pd.read_csv("../data/train.csv")
df = df[df["KPI ID"] == kpi_id]
values, labels, timestamps = df.value.values, df.label.values, df.timestamp.values
fig, ax = plt.subplots(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
colors = ['blue', 'red']
df["timestamp"] = df["timestamp"] - df.timestamp.values.min()


ax.scatter(df.timestamp.values, df.value.values,
           marker='.',
           c=df.label.values, cmap=clr.ListedColormap(colors))

plt.show()