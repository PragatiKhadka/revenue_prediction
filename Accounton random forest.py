import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_accounton.csv')
  
df1 = pd.read_csv(f)

X = df1[['ebit', 'total_liabilities', 'net_added_value', 
       'staff_costs', 'current_revenue', 'Large', 'Medium sized', 
       'Small', 'Very large', 'Antwerp', 'East-Flanders', 'Limburg', 
       'Vlaams Brabant', 'West-Flanders']]
y = df1['next_year_revenue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2)

from sklearn  import linear_model 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# Train the model
model = RandomForestRegressor(n_estimators = 10, random_state = 2)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

#Y_pred = model.predict(np.array([6.5]).reshape(1, 1))  # test the output by changing values

# Visualising the Random Forest Regression results
 
# arrange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
#X_grid = np.arange(min(df1[X]), max(df1[X]), 0.01)
 
# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value                 
#X_grid = X_grid.reshape((len(X_grid), 1))
 
# Scatter plot for original data
#plt.scatter(X, y, color = 'blue') 
 

# Model performance
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )

