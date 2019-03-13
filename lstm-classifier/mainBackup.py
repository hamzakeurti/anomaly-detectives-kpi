# %%
from Util import *
from extraction.Time import *
from visualization.visualize import *
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

logger = get_logger()
start_time = time.time()
unbeefed_train_data = load_train()

def setAnomolyToNan(dataframe):
    dataframe['value'] = dataframe.apply(lambda row: np.nan if row['label'] == 1 else row['value'],axis=1)
def getOnlyAnomolies(dataframe):
    values = dataframe["value"].values.copy()
    for i in range(len(values)):
        if (dataframe["label"][i] == 0):
            values[i] = np.nan
    return values

def fillNanTimesteps(df):
    dataframeRange = int(df.timestamp.values.max() / 60)
    # df_ = pd.DataFrame(columns=['timestamp', 'value', 'label', 'KPI ID'], index=np.arange(dataframeRange))
    timestep = 0
    dfCounter = 0
    higher = True
    # print(df_)
    countedDfCounter = False
    for i in range(dataframeRange):
        if (i*60 == df.timestamp.values[i+dfCounter]):
            countedDfCounter = False
        else :
            print(i*60 , " - ", df.timestamp.values[i+dfCounter])
            if (countedDfCounter == False):
                dfCounter+=1
            line = DataFrame({"timestamp": i * 60, "value": np.nan, "label": 0, "KPI ID": df["KPI ID"][0]}, index=[i])
            df = pd.concat([df.ix[:i], line, df.ix[i+1:]]).reset_index(drop=True)
            countedDfCounter = True
    return df

def fillNansValues(dataframe):
    for i in range(len(dataframe["value"].values)):
        if (np.isnan(dataframe["value"].values[i])):
            foundFour = False
            sum = 0
            iterator = -2
            numberPlussed = 0
            goBackwards = False
            while(foundFour == False):
                iteratorValue = i + iterator*1440
                if (iteratorValue < len(dataframe["value"].values) and iteratorValue > 0):
                    plussValue = dataframe["value"].values[iteratorValue]
                    if (np.isnan(plussValue) == False):
                        sum += plussValue
                        numberPlussed += 1
                    if (numberPlussed == 4):
                        foundFour = True
                if (goBackwards):
                    iterator -= 1
                else:
                    iterator += 1
                if (iterator > 8):
                    goBackwards = True

            dataframe["value"][i] = (float(sum))/4

        # else:
        #     print(dataframe["value"].values[i])

split = split_on_id(unbeefed_train_data)
for KPI_id, df in split.items():
    split[KPI_id] = df
    print(KPI_id)
    # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    # fig = visualize_anomalies(df.timestamp.values, df.value.values, df.label.values)
    fig, ax = plt.subplots(num=None, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')
    colors = ['blue', 'red']
    print(df)
    df["timestamp"] = df["timestamp"] - df.timestamp.values.min()
    newPlotValues = getOnlyAnomolies(df)
    getAnomolyData = setAnomolyToNan(df)
    df = fillNanTimesteps(df)
    fillNansValues(df)
    print("lets create csv!")
    df.to_csv(KPI_id + "filled.csv", sep='\t')
    ax.scatter(df.timestamp.values, df.value.values,
               marker='.',
               c=df.label.values, cmap=clr.ListedColormap(colors))

    plt.show()
    df.to_csv(KPI_id+"filled.csv", sep='\t')

    break