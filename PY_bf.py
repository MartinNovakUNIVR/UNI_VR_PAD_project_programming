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