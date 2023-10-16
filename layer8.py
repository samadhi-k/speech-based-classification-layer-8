#!/usr/bin/env python
# coding: utf-8

# ### Mount drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ### Import packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# ### Import data

# In[ ]:


train = pd.read_csv('drive/MyDrive/DataSets/Layer8/train.csv');
valid = pd.read_csv('drive/MyDrive/DataSets/Layer8/valid.csv');
test = pd.read_csv('drive/MyDrive/DataSets/Layer8/test.csv');


# ### Describe data

# In[ ]:


train.describe()


# In[ ]:


train.shape


# Train dataset has 28520 datapoints and 768 fetures

# In[ ]:


valid.shape


# Valid dataset has 750 datapoints

# In[ ]:


train


# ### Handle missing values

# In[ ]:


columns = train.columns
labels = columns[-4:]
features = columns[:-4]


# In[ ]:


train.isnull().sum()


# In[ ]:


valid.isnull().sum()


# In[ ]:


from sklearn.impute import SimpleImputer

# missing values are imputed with the median value
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(train[labels])
imp.fit(valid[labels])
train[labels] = imp.transform(train[labels])
valid[labels] =imp.transform(valid[labels])


# In[ ]:


train.isnull().sum()


# In[ ]:


valid.isnull().sum()


# In[ ]:


X_train = train[features]
Y_train = train[labels]
X_valid = valid[features]
Y_valid = valid[labels]


# ### Varience based feature selection

# In[ ]:


# scatter plot varience of features
plt.figure(figsize=(20,10))
plt.scatter(X_train.columns, X_train.var())


# Most of the feature varience is close to 0. Removing most of them will negatively imapact on the accuracy of the model

# In[ ]:


from sklearn.feature_selection import VarianceThreshold
threshold = VarianceThreshold(0.0001)
X_train_array = threshold.fit_transform(X_train)
X_train = pd.DataFrame(X_train_array, columns=X_train.columns)
X_train_array.shape, X_train.shape


# ### Correlation based feature selection

# In[ ]:


import seaborn as sns


# In[ ]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.9)
len(set(corr_features))


# In[ ]:


X_train = X_train.drop(corr_features, axis=1)
X_valid = X_valid.drop(corr_features, axis=1)


# 70 correlated features were dropped.

# ### Label 1 : Speaker ID

# In[ ]:


Y_train_1 = Y_train['label_1']
Y_valid_1 = Y_valid['label_1']


# In[ ]:


print(Y_train['label_1'].sort_values())


# In[ ]:


Y_train_1.head()
plt.figure(figsize=(20,10))
label_1 = Y_train['label_1']
label_1.value_counts().plot(kind='bar')


# In[ ]:


from sklearn.feature_selection import mutual_info_classif

mutual_info = mutual_info_classif(X_train, Y_train_1)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# removing features with less than 0.005 mutual information

# In[ ]:


from sklearn.feature_selection import SelectKBest

# selecting features with greater than 0.25 mutual information scores
X_train_1 = X_train.copy()
X_valid_1 = X_valid.copy()
feature_count_1 = mutual_info[mutual_info > 0.005].count()
sel_five_cols = SelectKBest(mutual_info_classif, k=feature_count_1)
sel_five_cols.fit(X_train_1, Y_train_1)
new_columns_1 = X_train_1.columns[sel_five_cols.get_support()]


# In[ ]:


print(f'{feature_count_1} features selected out of {len(X_train.columns)}')


# In[ ]:


X_train_1 = X_train_1[new_columns_1]
X_valid_1 = X_valid_1[new_columns_1]
X_train_1.shape, X_valid_1.shape


# Support Vector Machine

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


svm_1 = SVC()
param_grid = {'kernel': ('linear', 'poly', 'rbf')}
gs = GridSearchCV(svm_1, param_grid, cv=5)
gs.fit(X_train_1, Y_train_1)


# In[ ]:


def print_results(results, best_params) :
  for i in range(len(results)):
    print(f"Iteration: {results['params']} mean accuracy score: {results['mean_test_score']}")
  print(f"Best iteration {best_params}")


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_svm_1 = gs.best_estimator_


# K Nearest Neighbour

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

param_grid = {'n_neighbors': list(range(1,11))}
gs = GridSearchCV(knn_clf, param_grid, cv=5)
gs.fit(X_train_1, Y_train_1)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_knn_1 = gs.best_estimator_


# Random forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rf_1 = RandomForestClassifier()
param_grid = {'n_estimators': [100,200,300],'criterion':("gini", "entropy", "log_loss")}
gs = GridSearchCV(rf_1, param_grid, cv=5)
gs.fit(X_train_1, Y_train_1)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_rf_1 = gs.best_estimator_


# Best model is the SVM rbf model

# In[ ]:


output['label_1'] = best_svm_1.predict(X_test)


# In[ ]:


output.to_csv('output.csv')


# ### Label 2 : Age

# In[ ]:


Y_train_2 = Y_train['label_2']
Y_valid_2 = Y_valid['label_2']


# In[ ]:


print(Y_train_2.sort_values())


# In[ ]:


Y_train_2.head()
plt.figure(figsize=(20,10))
Y_train_2.value_counts().plot(kind='bar')


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, Y_train_2)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[ ]:


from sklearn.feature_selection import SelectKBest

# selecting features with greater than 0.01 mutual information scores
X_train_2 = X_train.copy()
X_valid_2 = X_valid.copy()
feature_count_2 = mutual_info[mutual_info > 0.01].count()
sel_five_cols = SelectKBest(mutual_info_classif, k=feature_count_2)
sel_five_cols.fit(X_train_2, Y_train_2)
new_columns_2 = X_train_2.columns[sel_five_cols.get_support()]


# In[ ]:


print(f'{feature_count_2} features selected out of {len(X_train.columns)}')


# In[ ]:


X_train_2 = X_train_2[new_columns_2]
X_valid_2 = X_valid_2[new_columns_2]
X_train_2.shape, X_valid_2.shape


# Support Vector Machine

# In[ ]:


svm_2 = SVC()
param_grid = {'kernel': ('linear', 'poly', 'rbf')}
gs = GridSearchCV(svm_2, param_grid, cv=5)
gs.fit(X_train_2, Y_train_2)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_svm_2 = gs.best_estimator_


# KNN

# In[ ]:


from sklearn.model_selection import GridSearchCV


knn_clf = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1,11))}
gs = GridSearchCV(knn_clf, param_grid, cv=5)
gs.fit(X_train_2, Y_train_2)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_knn_2 = gs.best_estimator_


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rf_2 = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300],'criterion':("gini", "entropy", "log_loss")}
gs = GridSearchCV(rf_2, param_grid, cv=5)
gs.fit(X_train_2, Y_train_2)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_rf_2 = gs.best_estimator_


# Best model is the KNN with with n_neighbour = 1

# In[ ]:


output['label_2'] = best_knn_2.predict(X_test)
output.to_csv('output.csv')


# ### Label 3 : Gender

# In[ ]:


Y_train_3 = Y_train['label_3']
Y_valid_3 = Y_valid['label_3']


# In[ ]:


Y_train_3.head()
plt.figure(figsize=(20,10))
Y_train_3.value_counts().plot(kind='bar')


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, Y_train_2)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X_train.columns
mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 8))


# In[ ]:


# selecting features with greater than 0.25 mutual information scores
X_train_3 = X_train.copy()
X_valid_3 = X_valid.copy()
feature_count_3 = mutual_info[mutual_info > 0.01].count()
sel_five_cols = SelectKBest(mutual_info_classif, k=feature_count_3)
sel_five_cols.fit(X_train_3, Y_train_3)
new_columns_3 = X_train_3.columns[sel_five_cols.get_support()]


# In[ ]:


print(f'{feature_count_3} features selected out of {len(X_train.columns)}')


# In[ ]:


X_train_3 = X_train_3[new_columns_3]
X_valid_3 = X_valid_3[new_columns_3]
X_train_3.shape, X_valid_3.shape


# In[ ]:


svm_3 = SVC()
param_grid = {'kernel': ('linear', 'poly', 'rbf')}
gs = GridSearchCV(svm_3, param_grid, cv=5)
gs.fit(X_train_3, Y_train_3)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_svm_3 = gs.best_estimator_


# KNN

# In[ ]:


from sklearn.model_selection import GridSearchCV


knn_3 = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1,11))}
gs = GridSearchCV(knn_3, param_grid, cv=5)
gs.fit(X_train_3, Y_train_3)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_knn_3 = gs.best_estimator_


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rf_3 = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300], 'criterion':("gini", "entropy", "log_loss")}
gs = GridSearchCV(rf_3, param_grid, cv=5)
gs.fit(X_train_3, Y_train_3)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_rf_3 = gs.best_estimator_


# Best model is the SVM linear model

# In[ ]:


output['label_3'] = best_svm_3.predict(X_test)
output.to_csv('output.csv')


# ### Label 4 : Accent

# In[ ]:


Y_train_4= Y_train['label_4']
Y_valid_4 = Y_valid['label_4']


# In[ ]:


plt.figure(figsize=(20,10))
Y_train_4.value_counts().plot(kind='bar')


# In[ ]:


# Apply SMOTE to address the data imbalance
from imblearn.over_sampling import SMOTE

X_train_4 = X_train.copy()
X_valid_4 = X_valid.copy()


# In[ ]:


smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_4, Y_train_4 = smote.fit_resample(X_train_4, Y_train_4)


# In[ ]:


plt.figure(figsize=(20,10))
Y_train_4.value_counts().plot(kind='bar')


# In[ ]:


svm_4 = SVC()
param_grid = {'kernal': ('linear', 'poly', 'rbf')}
gs = GridSearchCV(svm_4, param_grid, cv=5)
gs.fit(X_train_4, Y_train_4)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_svm_4 = gs.best_estimator_


# KNN

# In[ ]:


from sklearn.model_selection import GridSearchCV


knn_4 = KNeighborsClassifier()
param_grid = {'n_neighbors': list(range(1,11))}
gs = GridSearchCV(knn_4, param_grid, cv=5)
gs.fit(X_train_4, Y_train_4)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_knn_4 = gs.best_estimator_


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rf_4 = RandomForestClassifier()
param_grid = {'n_estimators': [100, 200, 300],'criterion':("gini", "entropy", "log_loss")}
gs = GridSearchCV(rf_4, param_grid, cv=5)
gs.fit(X_train_4, Y_train_4)


# In[ ]:


print_results(gs.cv_results_, gs.best_params_)
best_rf_4 = gs.best_estimator_


# Best model is the SVM rbf model

# In[ ]:


output['label_4'] = best_svm_4.predict(X_test)
output.to_csv('output.csv')

