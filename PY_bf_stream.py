#1) IMPORT LIBRARIES

#streamlit run c:/Users/novak/Documents/UNI_VR_PAD_project_programming/PY_bf_stream.py

import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.parquet as pq
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

#--------------------------------------------------------------------------NEW SECTION------------------------------------------------------------------------------------------------#

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
ID_bf_1_parquet_info = ID_bf_1.info()
print(ID_bf_1_parquet_info)

#print dataset's variables' basic summary stats
ID_bf_1_parquet_desc = ID_bf_1.describe()
print(ID_bf_1_parquet_desc)

#print dataset's variables' basic summary stats
ID_bf_1_parquet_corr = ID_bf_1.corr()
print(ID_bf_1_parquet_corr)

#Check columns data - categorical
print("Share of occurances per group - only categorical")   
def calculate_share_categorical(df):
    shares = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            value_counts = df[column].value_counts(normalize = True)
            shares[column] = value_counts
    return pd.DataFrame(shares)

ID_bf_1_parquet_share_cat = print(calculate_share_categorical(ID_bf_1_parquet))


#Streamlit 1 ---
st.title('Logit-Based Fraud Analysis and Prediction')

st.subheader('Is the Client a Frauder?')

st.write("""The aim of this project is to find the determinants of a bank fraud, and 
            infer a prediction. The user will be able to personally select among more
            than 70 variables and a dependent variable "frauder""")

st.subheader('1) Showing the initial dataset and its summary stats')

st.write("""------------------------------------------------------""")

st.sidebar.subheader('Chapter 1')
if st.sidebar.checkbox('DataFrame Complete'):
    st.write('*The DataFrame Complete*')
    st.write(ID_bf_1_parquet)
if st.sidebar.checkbox('DataFrame Complete - Summary Stats'):
    st.write('*The DataFrame Complete - Summary Stats*')
    st.write(ID_bf_1_parquet_desc)

st.write("""------------------------------------------------------""")

st.write("""After analyzing the dataset, I realized that there is too much data inside, 
            which slows down the whole process due to the slow internet at home. 
            Therefore, I will scale down the data set from 100K rows to less than 15K, 
            which is still considered a very large sample. 
            Also, since I want the data structure to remain constant,
            I will adjust the new dataset so that the ratio of groups in the "device_os"
            variable is the same (or almost) as before.""")
#Streamlit 1 ---

#--------------------------------------------------------------------------NEW SECTION------------------------------------------------------------------------------------------------#

#3) DATA PREPARATION & WRANGLING

##Reducing the dataset

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


#Streamlit 2 ---
st.subheader('2) Reducing the Dataset')

st.write("""Recognizing that the data set is too large, the goal of this section is to reduce it from 100K (rows) to 15K. 
            This will be done while keeping the same sample structure by the following specification.""")

st.write("""------------------------------------------------------""")

st.sidebar.subheader('Chapter 2')
if st.sidebar.checkbox('DataFrame Reduced'):
    st.write('*The DataFrame Reduced*')
    st.write(ID_bf_1_parquet_shorter)
    st.write("The target is 15000 rows, while the result is: " + str(len(ID_bf_1_parquet_shorter.fraud_bool)) + " which is acceptable.")

st.write("""------------------------------------------------------""")
#Streamlit 2 ---


##Create new variables

###Convert numerical variables into levels

Bank_fraud_df = ID_bf_1_parquet_shorter.copy()

#Create function to convert to 5 levels
def convert_to_levels(x, min_value, max_value):
    interval = (max_value - min_value) / 5
    if x < min_value + interval:
        return 1 #very low
    elif x < min_value + 2 * interval:
        return 2 #low
    elif x < min_value + 3 * interval:
        return 3 #medium
    elif x < min_value + 4 * interval:
        return 4 #high
    else:
        return 5 #very high

#Create a function to find min and max of a certain variable in a dataframe
def categorize_variable(dataframe, variable):
    min_value          = dataframe[variable].min()
    max_value          = dataframe[variable].max()
    dataframe['level'] = dataframe[variable].apply(lambda x : convert_to_levels(x, min_value, max_value))

#Create a function that nests previously created functions, calculates levels for numerical vars only and changes the label accordingly
def categorize_all_variables_in_dataframe(dataframe):
    numerical_vars = dataframe.select_dtypes(include = ['float64', 'int64']).columns #only numeric vars
    for variable in numerical_vars:
        categorize_variable(dataframe, variable)
        dataframe.rename(columns={'level': 'level_' + variable}, inplace = True) #change label level_"variable_name"

#Use the final function
categorize_all_variables_in_dataframe(Bank_fraud_df)

#Inspect the dataframe
print(Bank_fraud_df.info())

#Show results
print(Bank_fraud_df.head())


###Encode categorical variables into dummies

#Create function to encode categorical variables into dummies
def encode_dummies(dataframe):
    encode_dummies_cols = dataframe.select_dtypes(include = ['object']).columns #select categorical vars only
    for column in encode_dummies_cols:
        encode    = pd.get_dummies(dataframe[column], prefix = 'encoded_' + column) #encode dummies
        dataframe = dataframe.drop(column, axis = 1)
        dataframe = dataframe.join(encode)
    return dataframe

#Use the function
Bank_fraud_df = encode_dummies(Bank_fraud_df)

#Create a new parquet file
Bank_fraud_df.to_csv('Bank_fraud_df', sep = '\t', encoding = 'utf-8')

#Inspect the dataframe
print(Bank_fraud_df.info())

#Show results
print(Bank_fraud_df.head())


#Streamlit 3 ---
st.subheader('3) Creating New Variables')

st.write("""The data set is reasonably good, no null or missing values, and the variables are correctly labeled.
            Therefore, I decided to go ahead and artificially create new variables based on their levels and encoded dummies.
            The decision of whether to transform into levels or encoding dummies depends on the nature of the variable itself.""")

st.write("""------------------------------------------------------""")

st.write("""Numerical variables:
            The idea of converting them to levels is to reduce excessive variability in the data, but at the same time to convert them to a scale of 1 to 5 in order to expose the advantages of linearity.
            Obviously, not all variables converted to levels will be truly meaningful, as binary variables or other methodologically different variables have no meaning in levels.
            These will always will still be part of the function, but will not be used in the model.
            Categorical variables:
            Here I exploit the potential of python's function to get dummies,
            I create my own function to convert all categorical variables (objects) into dummies and
            label them accordingly to distinguish them consistently (like in the previous case).""")

st.sidebar.subheader('Chapter 3')
if st.sidebar.checkbox('DataFrame Reduced - New Variables'):
    st.write('*The DataFrame Reduced - New Variables*')
    st.write(Bank_fraud_df)

st.write("""------------------------------------------------------""")
#Streamlit 3 ---


#Streamlit 4 ---
st.subheader("4) Interesting Plots")

###Make ex-post summary statistics and plotting

####Adapt new dataframe for plotting

#Correlations:

#Subsetting based on each of the three variable supra-groups
Bank_fraud_df_original_vars = Bank_fraud_df.iloc[:,  0:26]
Bank_fraud_df_levels_vars   = Bank_fraud_df.iloc[:, 27:53]
Bank_fraud_df_dummies_vars  = Bank_fraud_df.iloc[:, 54:79]

#Computing the correlation per group
corr_original_vars = Bank_fraud_df_original_vars.corr().round(2)
corr_levels_vars   = Bank_fraud_df_levels_vars.corr().round(2)
corr_dummies_vars  = Bank_fraud_df_dummies_vars.corr().round(2)

#Histograms and boxplots:

#Subsetting and replacing values
Bank_fraud_df_reduced_plots = Bank_fraud_df[['customer_age', 'level_customer_age', 'level_credit_risk_score', 'level_income', 'level_days_since_request', 'fraud_bool']]

#Adapt the dataset for the plotting
Bank_fraud_df_reduced_plots['customer_age_range'] = np.where((Bank_fraud_df_reduced_plots['customer_age'] >=  0) & (Bank_fraud_df_reduced_plots['customer_age'] < 18), '<18',
                                                    np.where((Bank_fraud_df_reduced_plots['customer_age'] >= 18) & (Bank_fraud_df_reduced_plots['customer_age'] < 35), '18-34',
                                                    np.where((Bank_fraud_df_reduced_plots['customer_age'] >= 36) & (Bank_fraud_df_reduced_plots['customer_age'] < 50), '35-49',
                                                    np.where((Bank_fraud_df_reduced_plots['customer_age'] >= 50) & (Bank_fraud_df_reduced_plots['customer_age'] < 70), '50-69',
                                                    '>70'))))

Bank_fraud_df_reduced_plots.drop('customer_age', axis = 1, inplace = True)

Bank_fraud_df_reduced_plots.rename(columns = {'customer_age_range': 'customer_age'}, inplace = True)


####Correlation heatmaps
st.subheader("Correlation Heatmaps: Original, Levels, and Dummy Variables")

with st.expander("Show Correlation Heatmaps: Original, Levels, and Dummy Variables"):
    col_1, col_2 = st.columns(2)

    #corr_levels_vars
    with col_1:
        st.subheader('Correlation Between Level Variables')
        fig, ax = plt.subplots(figsize = (12, 10))
        sns.heatmap(corr_levels_vars, 
            cmap = "Spectral", 
            annot = False, linewidths = 0.01, linecolor = "black",
            vmin = -1.10, vmax = 1.10,
            square = True, ax = ax)
        st.pyplot(fig)

    #corr_dummies_vars
    with col_2:
        st.subheader('Correlation Between Dummy Variables')
        fig, ax = plt.subplots(figsize = (12, 10))
        sns.heatmap(corr_dummies_vars, 
            cmap = "Spectral", 
            annot = False, linewidths = 0.01, linecolor = "black",
            vmin = -1.10, vmax = 1.10,
            square = True, ax = ax)
        st.pyplot(fig)


####Histograms
st.subheader("Histograms: Customer Age, Credit Score, and Income Level by Fraud")

with st.expander("Show Histograms: Customer Age, Credit Score, and Income Level by Fraud"):
    col_4, col_5, col_6 = st.columns(3)

    #Level_income by fraud_bool
    with col_4:
        st.subheader('Histogram of Customer Age by Fraud')
        sns.histplot(x = "customer_age", hue = "fraud_bool", bins = np.arange(0, 6, 1), data = Bank_fraud_df_reduced_plots)
        plt.title('Histogram of Customer Age by Fraud', fontsize = 14)
        plt.xlabel('Customer age category', fontsize = 12)
        plt.ylabel('N° occurances', fontsize = 12)
        plt.legend(labels = ['Fraud', 'No Fraud'])
        plt.xticks(np.arange(0, 6, 1))
        plt.tight_layout()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

    #Level_credit_risk_score by fraud_bool
    with col_5:
        st.subheader('Histogram of Credit Score by Fraud')
        sns.histplot(x = "level_credit_risk_score", hue = "fraud_bool", bins = np.arange(0, 6, 1), data = Bank_fraud_df_reduced_plots)
        plt.title('Histogram of Credit Score by Fraud', fontsize = 14)
        plt.xlabel('Credit risk score levels (1 = very low, 5 = very high)', fontsize = 12)
        plt.ylabel('N° occurances', fontsize = 12)
        plt.legend(labels = ['Fraud', 'No Fraud'])
        plt.xticks(np.arange(0, 6, 1))
        plt.tight_layout()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)

    #level_income by fraud_bool
    with col_6:
        st.subheader('Histogram of Income Level by Fraud')
        sns.histplot(x = "level_income", hue = "fraud_bool", bins = np.arange(0, 6, 1), data = Bank_fraud_df_reduced_plots)
        plt.title('Histogram of Income Level by Fraud', fontsize = 14)
        plt.xlabel('Income levels (1 = very low, 5 = very high)', fontsize = 12)
        plt.ylabel('N° occurances', fontsize = 12)
        plt.legend(labels = ['Fraud', 'No Fraud'])
        plt.xticks(np.arange(0, 6, 1))
        plt.tight_layout()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)


####Boxplots
st.subheader("Boxplots: Credit Risk Score, Income, and Customer Age Level")

with st.expander("Show Boxplots: Credit Risk Score, Income, and Customer Age Level"):
    #Plotting the boxplots    
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    sns.boxplot(x = "fraud_bool", y = "level_credit_risk_score", data = Bank_fraud_df_reduced_plots, ax = axs[0])
    axs[0].set_title("Credit Risk Score Levels")
    sns.boxplot(x = "fraud_bool", y = "level_income", data = Bank_fraud_df_reduced_plots, ax = axs[1])
    axs[1].set_title("Income Levels")
    sns.boxplot(x = "fraud_bool", y = "level_customer_age", data = Bank_fraud_df_reduced_plots, ax = axs[2])
    axs[2].set_title("Customer Age Level")
    plt.suptitle("Boxplots for Credit Risk Score, Income and Customer Age Level")
    st.pyplot(fig)


####Log-scaled histograms
st.subheader("Log-Scaled Distributions of the Variables: days_since_request, name_email_similarity, and customer_age")

with st.expander("Show Log-Scaled Distributions of the Variables: days_since_request, name_email_similarity, and customer_age"):
    fig, axes = plt.subplots(3, 1, figsize = (8, 6), sharey = True, constrained_layout = True)
    fig.suptitle('Log-scaled distributions of days_since_request, name_email_similarity, and customer_age',
                 fontweight = "bold")
    sns.set_style('darkgrid')
    sns.histplot(Bank_fraud_df['days_since_request'], log_scale=True, ax=axes[0])
    sns.histplot(Bank_fraud_df['name_email_similarity'], log_scale=True, ax=axes[1])
    sns.histplot(Bank_fraud_df['customer_age'], log_scale=True, ax=axes[2])
    st.pyplot(fig)
#Streamlit 4 ---

#--------------------------------------------------------------------------NEW SECTION------------------------------------------------------------------------------------------------#

#4) MODEL DEVELOPMENT

#Streamlit 5 ---
st.subheader('5) Logit Model - Predict Frauds')

with st.expander('Show Logit Model'):
    model = LogisticRegression()
    x_choices = Bank_fraud_df.drop(['fraud_bool'], axis = 1)
    choices = st.multiselect('Select any [Combination of] Indipendent Variable/s:', list(x_choices.columns.values))
    y = Bank_fraud_df['fraud_bool']
    test_size = st.slider('Choose the Size of the Test Sample: ', min_value = 0.10, max_value = 0.90, step = 0.05)

    if len(choices) > 0 and st.button('CLICK TO RUN'):
        with st.spinner('Training...'):
            
            #set train/test data:
            x = Bank_fraud_df[choices]
            x_train, x_test, y_train, y_test = train_test_split(x, 
                                                                y,
                                                                test_size = test_size,
                                                                random_state = 2)

            #fit the model:
            model_fit = model.fit(x_train[choices], y_train)

            #predict and test:
            x_test = x_test.to_numpy().reshape(-1, len(choices))
            y_pred = model.predict(x_test)

            #model performance - scoring:
            Accuracy  = accuracy_score(y_test, y_pred)
            Precision = precision_score(y_test, y_pred)
            Recall    = recall_score(y_test, y_pred)
            #F1        = f1_score(y_test, y_pred)
            #AUC       = roc_auc_score(y_test, y_pred)

            st.write("Accuracy:  ", Accuracy)
            st.write("Precision: ", Precision)
            st.write("Recall:    ", Recall)
            #st.write("F1:        ", F1)
            #st.write("AUC:       ", AUC)

            #model performance - summary:
            #st.write(model_fit.summary())
#Streamlit 5 ---