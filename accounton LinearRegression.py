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
print(X.shape)


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

y_test = y_test["prediction"]=y_hat

"""
from sklearn.preprocessing import PolynomialFeatures  
poly_regs= PolynomialFeatures(degree= 2)  
x_poly= poly_regs.fit_transform(X)  
lin_reg_2 =linear_model.LinearRegression()  
lin_reg_2.fit(x_poly, y)  


plt.plot(X,y,color="blue")  
plt.plot(X, lin_reg_2.predict(poly_regs.fit_transform(X)), color="red")  
plt.title("Bluff detection model(Polynomial Regression)")  
plt.xlabel("Features")  
plt.ylabel("Revenue")  
#plt.show()  

plt.show()
"""
