import numpy as np
from pandas import DataFrame
import pandas as pd 

class Filler:
    @staticmethod
    def fillNansValues(dataframe, period = 1440):
        for i in range(len(dataframe["value"].values)):
            if (np.isnan(dataframe["value"].values[i])):
                foundFour = False
                sum = 0
                iterator = -2
                numberPlussed = 0
                goBackwards = False
                while (foundFour == False):
                    iteratorValue = i + iterator * period
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

                dataframe["value"][i] = (float(sum)) / 4
    
    @staticmethod
    def addNansToTheTimeseries(df, numberOfValues, kpiId, timestampInterval = 60, atTheBeginning = True):
        data = []
        for i in range(numberOfValues):
            data.insert(0, {"timestamp": (-(i + 1)) * 60, "value": np.nan, "label": 0, "KPI ID":kpiId})
        print(data)
        print(data)
        if (atTheBeginning):
            return pd.concat([pd.DataFrame(data), df], ignore_index=True)
        else:
            return pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        # return newDf