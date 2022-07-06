import json
import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt

# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_accounton.csv')
  
# returns JSON object as 
# a dictionary
#raw_data = json.load(f)

df1 = pd.read_csv(f)


X = df1[['ebit.2020', 'ebit.2019', 'ebit.2018', 'ebit.2017', 'ebit.2016',
       'ebit.2015', 'total_assets.2020',
       'total_assets.2019', 'total_assets.2018', 'total_assets.2017',
       'total_assets.2016', 'total_assets.2015', 'total_liabilities.2020',
       'total_liabilities.2019', 'total_liabilities.2018',
       'total_liabilities.2017', 'total_liabilities.2016',
       'total_liabilities.2015', 'operating_profit_and_loss.2020',
       'operating_profit_and_loss.2019', 'operating_profit_and_loss.2018',
       'operating_profit_and_loss.2017', 'operating_profit_and_loss.2016',
       'operating_profit_and_loss.2015', 'financial_profit_and_loss.2020',
       'financial_profit_and_loss.2019', 'financial_profit_and_loss.2018',
       'financial_profit_and_loss.2017', 'financial_profit_and_loss.2016',
       'financial_profit_and_loss.2015', 'staff_count.2020',
       'staff_count.2019', 'staff_count.2018', 'staff_count.2017',
       'staff_count.2016', 'staff_count.2015',
       'staff_costs.2020', 'staff_costs.2019', 'staff_costs.2018',
       'staff_costs.2016', 'staff_costs.2017', 'staff_costs.2015']]
y = df1[['revenue.2020', 'revenue.2019', 'revenue.2018', 'revenue.2017', 'revenue.2016', 'revenue.2015']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn  import linear_model 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

# Train the model
model = linear_model.LinearRegression()
model.fit(X, y)
y_hat = model.predict(X_test)

# Model performance
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Coefficient of determination (R^2): %.3f' % r2_score(y_test, y_hat))
print('Mean squared error (MSE): %.3f'% mean_squared_error(y_test, y_hat))
print('Root mean squared error (RMSE) : %.3f'% sqrt(mean_squared_error(y_test, y_hat)) )

from sklearn.preprocessing import PolynomialFeatures  
poly_regs= PolynomialFeatures(degree= 2)  
x_poly= poly_regs.fit_transform(X)  
lin_reg_2 =linear_model.LinearRegression()  
lin_reg_2.fit(x_poly, y)  

#plt.scatter(X,y,color="blue")  
#plt.plot(X, lin_reg_2.predict(poly_regs.fit_transform(X)), color="red")  
#plt.title("Bluff detection model(Polynomial Regression)")  
#plt.xlabel("Features")  
#plt.ylabel("Revenue")  
#plt.show()  

sns.residplot(data=df1, x='ebit.2018', y='revenue.2018', order=2)

plt.show()

