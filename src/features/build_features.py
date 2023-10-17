import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


#read data

df = pd.read_pickle("../../data/interim/02_outlier_removed_chauvenet.pkl")

df.isnull().sum()


plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

#Dealing with the missing values (Imputation)

predictor_columns = list(df.columns[:6])

#using Pandas interpolate method for imputation
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.isnull().sum()

#calculating the Avg duration of set

df[df['set']==25]['acc_y'].plot()
df[df['set']==50]['acc_y'].plot()

for s in df['set'].unique():
    start = df[df['set']==s].index[0]
    stop = df[df['set']==s].index[-1]
    
    duration = stop - start
    df.loc[(df['set']==s), "duration"] = duration.seconds
    
df.groupby(['category'])['duration'].mean()