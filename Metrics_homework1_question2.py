# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 20:30:06 2024

@author: lmackin_lbs
"""

import numpy as np
import math
import matplotlib.pyplot as plt
#from sklearn.ensemble import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

seed = 1


regressor = DecisionTreeRegressor(random_state = 1, max_depth=1, max_leaf_nodes=2,
                                  max_features=1)


def data_generation(n):
    x = np.random.normal(size = n)
    epsilon = np.random.normal(size = n)
    y = x*epsilon
    for i in range(0, len(x)):
        if x[i] > 0:
            y[i] += x[i]
        else:
            pass
    
    return x,y

x,y = data_generation(100000)


x = x[:, np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(x, y)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print ('MSE score:', MSE(y_test, y_pred))

X_grid = np.arange(np.min(x), np.max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure(figsize=(15,6))
plt.scatter(x, y, color = 'red', label='data')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green', label='Regression function')
plt.title('Decision Tree Regression')
plt.xlabel('x_values')
plt.ylabel('Target')
plt.legend()
plt.show()

#Part d: Graphing all the estimators
#Create function for CEF
def cef(x):
    cef = np.piecewise(x, [x <= 0, x > 0], [0, lambda x: x])
    return cef

def blp(x):
    blp = 1/np.sqrt(2*math.pi) + 0.5*x
    return blp
    
    

X_grid = np.arange(np.min(x), np.max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.figure(figsize=(15,6))
plt.scatter(x, y, color = 'red', label='data')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green', label='Regression function')
plt.plot(X_grid, cef(X_grid), color = "blue", label = "Conditional Expectation Function")
plt.plot(X_grid, blp(X_grid), color = "purple", label = "Best Linear Predictor")
plt.title('Plotting all predictors')
plt.xlabel('x_values')
plt.ylabel('Target')
plt.legend()
plt.show()
