import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt

# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_accounton.csv')
  
df1 = pd.read_csv(f)


X = df1[['ebit', 'total_liabilities', 'net_added_value', 
       'staff_costs', 'current_revenue', 'Large', 'Medium sized', 
       'Small', 'Very large', 'Antwerp', 'East-Flanders', 'Limburg', 
       'Vlaams Brabant', 'West-Flanders']]
y = df1[['next_year_revenue']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn  import linear_model 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# Train the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Model performance
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )

