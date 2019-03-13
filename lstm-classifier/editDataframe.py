import numpy as np
import pandas as pd

kpi_id = "02e99bd4f6cfb33f"
inputDim = 10
df = pd.read_csv("data/syntatic/02e99bd4f6cfb33ffilled.csv", sep='\t', index_col=0)
print(df)
