import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_accounton.csv')
  
df1 = pd.read_csv(f)

X = df1[['ebit', 'total_liabilities', 'net_added_value', 
       'staff_costs', 'current_revenue', 'Large', 'Medium sized', 
       'Small', 'Very large', 'Antwerp', 'East-Flanders', 'Limburg', 
       'Vlaams Brabant', 'West-Flanders']]
y = df1['next_year_revenue']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn  import linear_model 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# Train the model
model = RandomForestRegressor(n_estimators = 10, max_depth=7, random_state=0)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Model performance
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )


# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3)
# evaluate model
scores1 = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

# force scores to be positive
scores1 = np.absolute(scores1)
scores2 = np.absolute(scores2)
print('Coefficient of determination (R^2): %.3f (%.3f)' % (scores2.mean(), scores2.std()) )
print('Root mean squared error (RMSE): %.3f (%.3f)' % (scores1.mean(), scores1.std()) )
