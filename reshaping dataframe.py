import pandas as pd
import numpy as np
from pandas import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt


# Opening JSON file
f = open('/Users/Jorg/Accounton data/clean_accounton.csv')
  
df1 = pd.read_csv(f)

df_Large = df1.loc[df1['company_category']== "Large",:]
df_Medium = df1.loc[df1['company_category']== 'Medium sized',:]
df_Small = df1.loc[df1['company_category']== 'Small',:]
df_Very_large= df1.loc[df1['company_category']== 'Very large',:]

reshape_df = pd.DataFrame()
full_df = pd.DataFrame()
years = ['2015', '2016', '2017', '2018', '2019']
Features = ['ebit', 'ebitda' , 'profit_and_loss_after_taxes' , 'total_assets' , 'total_liabilities' , 'operating_profit_and_loss' , 'financial_profit_and_loss' , 'staff_count' , 'net_added_value' , 'staff_costs']

Total_clean_df = pd.DataFrame()

for y in years:
                reshape_df['vat_number'] = df1['vat_number']
                reshape_df['company_name'] = df1['company_name']
                reshape_df['company_category'] = df1['company_category']
                reshape_df['province'] = df1['province']
                reshape_df['Year_of_P'] = y
                for f in Features:
                    reshape_df[f"{f}"] = df1[f"{f}.{y}"]
                reshape_df[f"current_revenue"] = df1[f"revenue.{y}"]
                reshape_df[f"next_year_revenue"] = df1[f"revenue.{str(int(y)+1)}"]
                
                full_df = pd.concat([full_df,reshape_df], axis=0)

full_df = full_df.sort_values(by=['vat_number','Year_of_P'])
print(full_df)

years = ['2015', '2016','2017','2018','2019']
Features = ['ebit', 'ebitda' , 'profit_and_loss_after_taxes' , 'total_assets' , 'total_liabilities' , 'operating_profit_and_loss' , 'financial_profit_and_loss' , 'staff_count', 'revenue' , 'net_added_value' , 'staff_costs']
for feature in Features:
    for year in years:
        df_Large[f'{feature}.{year}'][df_Large[f'{feature}.{year}'].isna()] = df_Large[f'{feature}.{year}'].mean()
        df_Medium[f'{feature}.{year}'][df_Medium[f'{feature}.{year}'].isna()] = df_Medium[f'{feature}.{year}'].mean()
        df_Small[f'{feature}.{year}'][df_Small[f'{feature}.{year}'].isna()] = df_Small[f'{feature}.{year}'].mean()
        df_Very_large[f'{feature}.{year}'][df_Very_large[f'{feature}.{year}'].isna()] = df_Very_large[f'{feature}.{year}'].mean()

all_df1 = [df_Large,df_Medium,df_Small, df_Very_large ]
for each_df in all_df1:
    Total_clean_df = pd.concat([full_df,each_df])
print(Total_clean_df)

Total_clean_df.to_csv('/Users/Jorg/Accounton data/clean_reshaped_accounton.csv', index=False)

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

sns.set(rc={'figure.figsize':(25,15)})

# creating the correlation dataset using Pearson method (linear relation)
pc = Total_clean_df[n_variables].corr(method ='pearson')

cols = n_variables
ax = sns.heatmap(pc, annot=True,
                 yticklabels=cols,
                 xticklabels=cols,
                 annot_kws={'size':10},
                 cmap="Greens")

plt.show()