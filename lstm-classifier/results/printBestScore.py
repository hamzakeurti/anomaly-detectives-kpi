import pandas as pd
import pickle
import matplotlib
import numpy
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

bestFScoreKpi = {}
configKPI = {}
for line in open("lstmModel"):
    parts = line.rstrip("\n").split(", ")
    kpi = parts[0].split(" = ")[-1]
    fscore = float(parts[1].split(" = ")[-1])
    absTresh = parts[-1].split(" = ")[-1]
    absTresh = absTresh[2:-3]
    iteration = float(parts[2].split(" = ")[-1])
    # print(fscore)
    if (kpi not in bestFScoreKpi.keys()):
        bestFScoreKpi[kpi] = 0
        configKPI[kpi] = {"iteration" : 0, "threshold":[]}
    if (bestFScoreKpi[kpi] < fscore):
        bestFScoreKpi[kpi] = fscore
        configKPI[kpi]["threshold"].append(str(absTresh))
        configKPI[kpi]["iteration"] = iteration + 1

# with open('lstmConfig.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(configKPI, f)
lines = open("resultsCombinedDonut.txt").readlines()

xd = []
yd = []
y = []
donutDict = {}
for line in lines:
    parts = line.rstrip("\n").split(" ")
    kpi = parts[2]
    score = parts[5]
    xd.append(kpi[:-(len(kpi)-2)])
    yd.append(float(score[:-1]))
    donutDict[kpi] = score

print(sorted(bestFScoreKpi.items(), key=lambda s: s[0]))
print(sorted(donutDict.items(), key=lambda s: s[0]))
xd = []
yd = []
y = []
for key in donutDict.keys():
    yd.append(float(donutDict[key][:-1]))
    y.append(bestFScoreKpi[key])
    xd.append(key[:-(len(key)-2)])

# for kpiId in bestFScoreKpi.keys():
#


import numpy as np
import matplotlib.pyplot as plt

N = len(xd)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, y, width, color='royalblue')
rects2 = ax.bar(ind+width, yd, width, color='seagreen')

# add some
ax.set_ylabel('Scores')
ax.set_title('Score of LSTM and Donut')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels( (xd) )

ax.legend( (rects1[0], rects2[0]), ('LSTM', 'Donut') )

plt.show()

