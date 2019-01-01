

def loadConfig3():
    baseDir = "/Users/LarsErik/Skole/tsinghua/fag/anm/project/classifiers/lstm-classifier/"
    lines = open(baseDir + "results/lstmModel3").readlines()
    config = {}
    for line in lines:
        split = line.rstrip("\n").split("=")
        config[split[0]] = split[1][2:-2]
    return config


print(loadConfig3())
