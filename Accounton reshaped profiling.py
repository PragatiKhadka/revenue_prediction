import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sn
import matplotlib.pyplot as plt
#from markupsafe import soft_unicode
from pandas_profiling import ProfileReport

# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_reshaped_accounton.csv')
  
df1 = pd.read_csv(f)

  
#nan_value = float("NaN")
#df1.replace(0, nan_value, inplace=True)
#df1.replace("", nan_value, inplace=True)
  
#df1.dropna(how='all', axis=1, inplace=True)

df_category = pd.get_dummies(df1['company_category'])
df_new = pd.concat([df1, df_category], axis=1)
df_province = pd.get_dummies(df1['province'])
df_new = pd.concat([df_new, df_province], axis=1)
df_new

sn.set(rc={'figure.figsize':(40,40)})

n_variables=['ebit.2020', 'ebit.2019', 'ebit.2018', 'ebit.2017', 'ebit.2016',
       'ebit.2015', 'total_liabilities.2020',
       'total_liabilities.2019', 'total_liabilities.2018',
       'total_liabilities.2017', 'total_liabilities.2016',
       'total_liabilities.2015', 'revenue.2020', 'revenue.2019',
       'revenue.2018', 'revenue.2017', 'revenue.2016', 'revenue.2015',
       'net_added_value.2020', 'net_added_value.2019', 'net_added_value.2018',
       'net_added_value.2017', 'net_added_value.2016', 'net_added_value.2015',
       'staff_costs.2020', 'staff_costs.2019', 'staff_costs.2018',
       'staff_costs.2016', 'staff_costs.2017', 'staff_costs.2015']

profile = ProfileReport(df1[n_variables], title="Pandas Profiling Report")

profile.to_file("Profiling_crop_reshaped.html")
