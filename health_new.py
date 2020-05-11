# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:53:39 2020 
  
@author: Tanvi saxena  
"""  
#importing libraries
  
import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd

#importing dataset

dataset=pd.read_csv('health_insurance.csv')  #dataset is a variable declared to store the dataset

#importing the independant variable matrix ie the matrix of Features

X=dataset.iloc[:,:-1].values

#importing the dependant variable vector(column vector)

y=dataset.iloc[:,6].values 
 
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,4,5])], remainder='passthrough')  #OneHotEncoder makes columns depending on the no of categories or features inthe dataset


X = np.array(ct.fit_transform(X), dtype=np.float) #1st column is California;compare with the dataset

#AVOIDING THE DUMMY VARIABLE TRAP ie removing one dummy variable(Just for intuition sake the code below is given however we need
# we need not do it everytime since the library is taking care of that)


X=X[:,1:] #remember the rule that we exclude 1 dummy variable ie in the multilinear equation all dummy variables and constant cannot be included hence we exclude ine dummy variable


# Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/10, random_state = 0)

#learning;fitting the MLR to training set 

from sklearn.linear_model import LinearRegression 

regressor=LinearRegression()# creating an object of class LinearRegression which further calls it

regressor.fit(X_train,y_train)#where do we fit this object; ans:to the training set of the dataset. 

#Predicting test set results ;storing predicted values in vector called y_pred

""" and comparing predicted profits with real profits.;here regressor acts like the ml model which has 
learnt the relations."""

y_pred=regressor.predict(X_test) 

#Backward Elimination

import statsmodels.api as sm
"""adding x0=1 column to the dataset"""  
X=np.append(arr=np.ones((1200,1)).astype(int),values=X,axis=1) 

#backward elimination

#step 2:fit the full model with all possible predictors

X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10]] 
#we are fitting multilinear regression model from a differnt classcalled OLS to X_opt

regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 

#step3:How do we look at the predictor with highest p values?

regressor_OLS.summary()  
""" 
The lower the p value, the more signficant your independant variable with respect to 
your dependant variable"""

#Step 4: Remove the predictor; Refer X and modify x1 and x2 x3...

X_opt=X[:,[1,2,3,4,5,6,7,8,9,10]]                #remove 0
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#repeat 
X_opt=X[:,[1,3,4,5,6,7,8,9,10]]                  #remove 2
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_OLS.summary()
#repeat
X_opt=X[:,[1,4,5,6,7,8,9,10]]                    #remove 3
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_OLS.summary()
#repeat
X_opt=X[:,[1,4,5,6,7,8,9]]                      
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_OLS.summary()
#repeat
X_opt=X[:,[4,5,6,7,8,9]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_OLS.summary()