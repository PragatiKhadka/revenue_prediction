import json
import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sn
import matplotlib.pyplot as plt
#from markupsafe import soft_unicode
from pandas_profiling import ProfileReport

# Opening JSON file
f = open('/Users/Jorg/Accounton data/accounton_data.json')
  
# returns JSON object as 
# a dictionary
raw_data = json.load(f)

df = pd.json_normalize(raw_data)

df1 = df[df['ebit.2019'].notna()]

sn.set(rc={'figure.figsize':(40,40)})

n_variables=['ebit.2020', 'ebit.2019', 'ebit.2018', 'ebit.2017', 'ebit.2016',
       'ebit.2015', 'ebitda.2020', 'ebitda.2019', 'ebitda.2018', 'ebitda.2017',
       'ebitda.2016', 'ebitda.2015', 'profit_and_loss_after_taxes.2020',
       'profit_and_loss_after_taxes.2019', 'profit_and_loss_after_taxes.2018',
       'profit_and_loss_after_taxes.2017', 'profit_and_loss_after_taxes.2016',
       'profit_and_loss_after_taxes.2015', 'total_assets.2020',
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
       'staff_count.2016', 'staff_count.2015', 'revenue.2020', 'revenue.2019',
       'revenue.2018', 'revenue.2017', 'revenue.2016', 'revenue.2015',
       'net_added_value.2020', 'net_added_value.2019', 'net_added_value.2018',
       'net_added_value.2017', 'net_added_value.2016', 'net_added_value.2015',
       'staff_costs.2020', 'staff_costs.2019', 'staff_costs.2018',
       'staff_costs.2016', 'staff_costs.2017', 'staff_costs.2015']

profile = ProfileReport(df1[n_variables], title="Pandas Profiling Report")

profile.to_file("Profiling_crop.html")
