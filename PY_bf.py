#1) IMPORT LIBRARIES

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve



#2) IMPORT DATASETS AND INITIAL DATA EXPLORATION

## Importing dataset

#Read csv
ID_bf_1 = pd.read_csv('ID_bf_1.csv')

#Convert to parquet to increase efficiency (reduce data "weight")
ID_bf_1.to_parquet('output.parquet', engine = 'pyarrow')
ID_bf_1_parquet = pd.read_parquet('output.parquet', engine = 'pyarrow')
ID_bf_1_parquet = pd.DataFrame(ID_bf_1_parquet)
print(ID_bf_1_parquet.head())


## Make preliminary inspections

#print dataset's variables' basic information
print(ID_bf_1.info())

#print dataset's variables' basic summary stats
print(ID_bf_1.describe())

#print dataset's variables' basic summary stats
print(ID_bf_1.corr())

#check columns data
'''print("N° occurances per group")
for col in ID_bf_1_parquet.columns:
    print(col)
    print(ID_bf_1_parquet[col].value_counts())
    print("\n")'''

#Check columns data - categorical
print("Share of occurances per group - only categorical")   
def calculate_share_categorical(df):
    shares = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            value_counts = df[column].value_counts(normalize = True)
            shares[column] = value_counts
    return pd.DataFrame(shares)

print(calculate_share_categorical(ID_bf_1_parquet))

'''After analyzing the dataset, I realized that there is too much data inside, which slows down the whole process due to the slow internet at home. Therefore, I will scale down the data set from 1M rows to less than 100K, which is still considered a very large sample. Also, since I want the data structure to remain constant, I will adjust the new dataset so that the ratio of groups in the "device_os" variable is the same (or almost) as before.'''

