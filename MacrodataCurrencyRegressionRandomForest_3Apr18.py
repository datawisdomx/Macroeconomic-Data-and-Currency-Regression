#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 12:21:10 2018

@author: nitinsinghal
"""

# Macrodata Currency Regression Tests - Multiple, Polynomial. Random Forest tests.
# Please refer the supporting document MacroCcyRegressionRandomForestResultsSummary_3Apr18.docx for details

#Import libraries
#import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the macro and ccy data
# Make sure you change the file depending on currency pair you are evaluating
# US data is available in the same file so you can easily change column range to 
# set the regression data

macroccydata = pd.read_csv('/Users/nitinsinghal/DropboxNew/Dropbox/MachineLearning/PublishedResearch/MacroCcyRegression2018/macroccyeurusdalldata_3Apr18.csv')
X = macroccydata.iloc[:, 2:25].values
y = macroccydata.iloc[:, 1].values

# 6 month lag, 12 month lag, etc between ccy price and macro data can be done 
# by manipulating the source file or the row range as required


# Splitting the dataset into the Training set and Test set
# You can change the test_size and other parameters to get best fit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#multlin_reg = LinearRegression()
#multlin_reg.fit(X_train, y_train)

# Predicting a new result with Multiple Linear Regression
#y_pred = multlin_reg.predict(X_test)


# Fitting Polynomial Regression to the dataset - NOT USEFUL
#from sklearn.preprocessing import PolynomialFeatures

# Fitting Random Forest Regression to the dataset
# You can change the n_estimators and other parameters to get best fit
from sklearn.ensemble import RandomForestRegressor
randomforest_reg = RandomForestRegressor(n_estimators = 20, random_state = 0)
randomforest_reg.fit(X_train, y_train)

# Predicting a new result with Random Forest Regression
y_pred = randomforest_reg.predict(X_test)

# Check the importance of each feature
imp = randomforest_reg.feature_importances_
nfeat = randomforest_reg.n_features_
# Quality of predictions - MSE,MAE, R2. MSE measures square of difference between estimated and actual values
# R2 measures how well future samples are likely to be predicted by the model, max 1
mse = mean_squared_error(y_test , y_pred)
mae = mean_absolute_error(y_test , y_pred)
r2 = r2_score(y_test , y_pred)