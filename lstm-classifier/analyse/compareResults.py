import pandas as pd

leDf = pd.read_csv("submission_le.csv")
leDf2 = pd.read_csv("submission_7.csv")
kuDf = pd.read_csv("submission_le.csv")


# print(len(leDf2.timestamp.values))
# print(len(leDf2.value.values))
for kpi_id in set (leDf["KPI ID"]):
    kuFiltredDf = kuDf[kuDf["KPI ID"] == kpi_id]
    leFiltredDf = leDf[leDf["KPI ID"] == kpi_id]
    le2FiltredDf = leDf2[leDf2["KPI ID"] == kpi_id]
    leFiltredsum, kuFilredSum = leFiltredDf[leFiltredDf.predict == 1].sum()["predict"], kuFiltredDf[kuFiltredDf.predict == 1].sum()["predict"]
    le2Filsum = le2FiltredDf[le2FiltredDf.predict == 1].sum()["predict"]
    print("kpi = {}, leFilteredSum = {}, kuFilteredSum = {}, le2 = {}".format(kpi_id, leFiltredsum, kuFilredSum, le2Filsum))