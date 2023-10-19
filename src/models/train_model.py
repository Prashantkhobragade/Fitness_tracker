import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

from LearningAlgorithms import ClassificationAlgorithms

#plot settings
plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

#load the data

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

#create a training and test set

df_train = df.drop(['participent','category','set'], axis=1)

X = df_train.drop("label", axis=1)
y = df_train['label']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)


fig, ax = plt.subplots(figsize=(10,5))
df_train['label'].value_counts().plot(
    kind="bar", ax=ax, color="Lightblue", label="Total"
)
y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label='Train')
y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label='Test')
plt.legend()
plt.show()


#split feature subset

basic_features = ["acc_x","acc_y","acc_z", "gyr_x","gry_y","gry_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
freq_features = [f for f in df_train.columns if (("_freq" in f) or ("_pse" in f))]
cluster_features = ['cluster']

print("Basic features:", len(basic_features))
print("Square features:", len(square_features))
print("PCA features:", len(pca_features))
print("Time features:", len(time_features))
print("Frequency features:", len(freq_features))
print("Cluster features:", len(cluster_features))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

# Perfrom forward feature selection using simple Decision Tree
