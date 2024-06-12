# -*- coding: utf-8 -*-
"""
@author: from dzone blog
"""

# we import the necessary frameworks
import pandas as pd
import numpy as np

# we create dummy data to work with
data = {'A': [1, 2, None, 4], 'B': [5, None, None, 8], 'C': [10, 11, 12, 13]}

# we create and print the dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# TECHNIQUE: ROW REMOVAL > we remove rows with any missing values
df_cleaned = df.dropna()
print("Row(s) With Null Value(s) Deleted:\n" + str(df_cleaned), "\n")

# TECHNIQUE: COLUMN REMOVAL -> we remove columns with any missing values
df_cleaned_columns = df.dropna(axis=1)
print("Column(s) With Null Value(s) Deleted:\n" + str(df_cleaned_columns), "\n")

#%%
# IMPUTATION
# we create dummy data to work with
data = {'A': [1, 2, None, 4], 'B': [5, None, None, 8], 'C': [10, 11, 12, 13]}

# we create and print the dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we impute the missing values with mean
df['A'] = df['A'].fillna(df['A'].mean())
df['B'] = df['B'].fillna(df['B'].median())
print("DataFrame After Imputation:\n" + str(df), "\n")

#%%
# SMOOTHING
# we create dummy data to work with
data = {'A': [1, 2, None, 4],
        'B': [5, None, None, 8],
        'C': [10, 11, 12, 13]}

# we create and print the dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we calculate the moving average for smoothing
df['A_smoothed'] = df['A'].rolling(window=2).mean()
print("Smoothed Column A DataFrame:\n" + str(df), "\n")

#%%
# BINNING
# we create dummy data to work with
data = {'A': [1, 2, None, 4],
        'B': [5, None, None, 8],
        'C': [10, 11, 12, 13]}

# we create and print the dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we bin the data into discrete intervals
bins = [0, 5, 10, 15]
labels = ['Low', 'Medium', 'High']

# we apply the binning on column 'C'
df['Binned'] = pd.cut(df['C'], bins=bins, labels=labels)

print("DataFrame Binned Column C:\n" + str(df), "\n")

#%%
# NORMALIZATION
# we import the necessary frameworks
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# we create dummy data to work with
data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we apply mix-max normalization to our data using sklearn
scaler = MinMaxScaler()

df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Normalized DataFrame:\n" + str(df_normalized), "\n")

#%%
# STANDARDIZATION
# we create dummy data to work with
data = {'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we import 'StandardScaler' from sklearn
from sklearn.preprocessing import StandardScaler

# we apply standardization to our data
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print("Standardized DataFrame:\n" + str(df_standardized), "\n")

#%%
# ONE-HOT ENCODING
# we import the necessary framework
from sklearn.preprocessing import OneHotEncoder

# we create dummy data to work with
data = {'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we apply one-hot encoding to our categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['Color']])

encoded_df = pd.DataFrame(encoded_data,
                          columns=encoder.get_feature_names_out(['Color']))
print("OHE DataFrame:\n" + str(encoded_df), "\n")

#%%
# LABEL ENCODING
# we import the necessary framework
from sklearn.preprocessing import LabelEncoder

# we create dummy data to work with
data = {'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we apply label encoding to our dataframe
label_encoder = LabelEncoder()
df['Color_encoded'] = label_encoder.fit_transform(df['Color'])
print("Label Encoded DataFrame:\n" + str(df), "\n")

#%%
# CORRELATION MATRIX
# we import the necessary frameworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# we create dummy data to work with
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [5, 4, 3, 2, 1]}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we compute the correlation matrix of our features
correlation_matrix = df.corr()

# we visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

#%%
# CHI-SQUARE STATISTIC
# we import the necessary frameworks
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# we create dummy data to work with
data = {'Feature1': [1, 2, 3, 4, 5],
        'Feature2': ['A', 'B', 'A', 'B', 'A'],
        'Label': [0, 1, 0, 1, 0]}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we encode the categorical features in our dataframe
label_encoder = LabelEncoder()
df['Feature2_encoded'] = label_encoder.fit_transform(df['Feature2'])

print("Encocded DataFrame:\n" + str(df), "\n")

# we apply the chi-square statistic to our features
X = df[['Feature1', 'Feature2_encoded']]
y = df['Label']
chi_scores = chi2(X, y)
print("Chi-Square Scores:", chi_scores)

#%%
# PRINCIPAL COMPONENT ANALYSIS
# we import the necessary framework
from sklearn.decomposition import PCA

# we create dummy data to work with
data = {'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [5, 4, 3, 2, 1]}

# we print the original dataframe for viewing
df = pd.DataFrame(data)
print("Original DataFrame:\n" + str(df), "\n")

# we apply PCA to our features
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=['PC1', 'PC2'])

# we print the dimensionality reduced features
print("PCA Features:\n" + str(df_pca), "\n")
