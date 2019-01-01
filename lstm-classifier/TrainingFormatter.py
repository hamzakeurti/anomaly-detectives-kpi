import numpy as np
from Filler import Filler


class TrainingFormatter:
    @staticmethod
    def predictOneTimestepAhed(inputDim, values, labels, timestamps):
        as_strided = np.lib.stride_tricks.as_strided

        x = as_strided(values, (len(values), inputDim), values.strides * 2)[:-(inputDim)]
        y, t = values[(inputDim):], timestamps[(inputDim):]
        return x,y,t

    @staticmethod
    def predictOneAnomolyAhed(inputDim, values, labels, timestamps):
        as_strided = np.lib.stride_tricks.as_strided

        x = as_strided(values, (len(values), inputDim), values.strides * 2)[:-(inputDim-1)]
        y, t = labels[(inputDim - 1):], timestamps[(inputDim - 1):]
        return x, y, t

    @staticmethod
    def strideWithMissingInMiddle(values, inputDim):
        returnArray = []
        middleIndex = int(inputDim/2)
        for i in range(len(values)-(inputDim-1)):
            innerArray = []
            for l in range(inputDim):
                if (l != middleIndex):
                    innerArray.append(values[i+l])
            returnArray.append(np.array(innerArray))
        return np.array(returnArray)

    @staticmethod
    def predictTimestampInBetween(inputDim, values, labels, timestamps):
        strideWindow = int(inputDim/2)
        x = TrainingFormatter.strideWithMissingInMiddle(values, inputDim)
        y, t = values[strideWindow:-strideWindow], timestamps[strideWindow:-strideWindow]
        return x,y,t
