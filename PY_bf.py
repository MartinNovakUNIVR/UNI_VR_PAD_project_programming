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


#3) DATA PREPARATION & WRANGLING

##Reducing the dataset

'''Recognizing that the data set is too large, the goal of this section is to reduce it from 1M (rows) to 100k. This will be done while keeping the same sample structure by the following specification.'''

#Creating separate dataframes
ID_bf_1_parquet_linux     = ID_bf_1_parquet[ID_bf_1_parquet["device_os"] == "linux"]
ID_bf_1_parquet_macintosh = ID_bf_1_parquet[ID_bf_1_parquet["device_os"] == "macintosh"]
ID_bf_1_parquet_other     = ID_bf_1_parquet[ID_bf_1_parquet["device_os"] == "other"]
ID_bf_1_parquet_windows   = ID_bf_1_parquet[ID_bf_1_parquet["device_os"] == "windows"]
ID_bf_1_parquet_x11       = ID_bf_1_parquet[ID_bf_1_parquet["device_os"] == "x11"]

#Calculate the target number of rows
num_rows_linux     = int(15000 * 0.343673)
num_rows_macintosh = int(15000 * 0.051181)
num_rows_other     = int(15000 * 0.335533)
num_rows_windows   = int(15000 * 0.265053)
num_rows_x11       = int(15000 * 0.004560)

#Select the rows within each dataframe at random
rows_linux     = np.random.choice(ID_bf_1_parquet_linux.index.values, num_rows_linux, replace = False)
rows_macintosh = np.random.choice(ID_bf_1_parquet_macintosh.index.values, num_rows_macintosh, replace = False)
rows_other     = np.random.choice(ID_bf_1_parquet_other.index.values, num_rows_other, replace = False)
rows_windows   = np.random.choice(ID_bf_1_parquet_windows.index.values, num_rows_windows, replace = False)
rows_x11       = np.random.choice(ID_bf_1_parquet_x11.index.values, num_rows_x11, replace = False)

#Create the new reduced df by appending previously created ones
ID_bf_1_parquet_shorter = pd.concat([ID_bf_1_parquet_linux.loc[rows_linux],
                                     ID_bf_1_parquet_macintosh.loc[rows_macintosh],
                                     ID_bf_1_parquet_other.loc[rows_other],
                                     ID_bf_1_parquet_windows.loc[rows_windows],
                                     ID_bf_1_parquet_x11.loc[rows_x11]])

#Show results
print(ID_bf_1_parquet_shorter.head())
print("The target is 15000 rows, while the result is: " + str(len(ID_bf_1_parquet_shorter.fraud_bool)) + " which is acceptable.")


