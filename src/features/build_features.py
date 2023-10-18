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
    
duration_df = df.groupby(['category'])['duration'].mean()

#for heavy sets
duration_df.iloc[0]/5  

#for medium set
duration_df.iloc[1]/10

#Butterworth low-pass filter

df_lowpass = df.copy()
LowPass = LowPassFilter()

fs = 1000/200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass['set']==45]
print(subset['label'][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor = (0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor = (0.5, 1.15), fancybox=True, shadow=True)


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    


## Principal Component Analysis

df_pca = df_lowpass.copy()

PCA =PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

#using Elbow Method 
plt.figure(figsize=(10,10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("principal component number")
plt.ylabel("explained variable")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


## Sum Of Squred Attribute

df_squred = df_pca.copy()

acc_r = df_squred['acc_x'] ** 2 + df_squred['acc_y'] ** 2 +df_squred['acc_z'] ** 2
gyr_r = df_squred['gyr_x'] ** 2 + df_squred['gyr_y'] ** 2 +df_squred['gyr_z'] ** 2

df_squred['acc_r'] = np.sqrt(acc_r)
df_squred['gyr_r'] = np.sqrt(gyr_r)


