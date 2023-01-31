#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#Import the data set from Desktop
dataset = pd.read_csv('M_Regression.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.30, random_state=0)

#regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#for predict the test values
y_prdict=reg.predict(X_test)
print(y_prdict);

#Visualize the Traing data

# Performance Evaluation
# 1. Calculate the Mean Square Error

mse = mean_squared_error(y_test, y_prdict)
print("Mean Squared Error:", mse)

# 2. Root Mean Square Error 
rmse = np.sqrt(mse)
print("Mean Squared Error:", rmse)

# 3.Calculate mean absolute error
mae = mean_absolute_error(y_test, y_prdict)
print("Mean Absolute Error:", mae)


