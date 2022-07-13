#importing libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt

# opening the datafile
with open('/Users/Jorg/Accounton data/clean_accounton.csv') as f:
    # read the csv file
    df = pd.read_csv(f)

# choosing the features (X)
X = df[['ebit', 'total_liabilities', 'net_added_value', 
       'staff_costs', 'current_revenue', 'Large', 'Medium sized', 
       'Small', 'Very large', 'Antwerp', 'East-Flanders', 'Limburg', 
       'Vlaams Brabant', 'West-Flanders']]

# choosing the target (y)
y = df[['next_year_revenue']]

# splitting X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# Train the model
model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8, random_state=7)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)

# Model performance
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )

# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate model
scores1 = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

# force scores to be positive
scores1 = np.absolute(scores1)
scores2 = np.absolute(scores2)
print('Coefficient of determination (R^2): %.3f (%.3f)' % (scores2.mean(), scores2.std()) )
print('Root mean squared error (RMSE): %.3f (%.3f)' % (scores1.mean(), scores1.std()) )
