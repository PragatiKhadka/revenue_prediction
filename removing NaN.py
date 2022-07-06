import json
import pandas as pd
import numpy as np
from pandas import json_normalize

# Opening JSON file
f = open('/Users/Jorg/Accounton data/accounton_data.json')
  
# returns JSON object as 
# a dictionary
raw_data = json.load(f)

df = pd.json_normalize(raw_data)

df1 = df[df['ebit.2019'].notna()]

# splitting the data frame based on 'company_category'
df_Large = df1.loc[df1['company_category']== "Large",:]
df_Medium = df1.loc[df1['company_category']== 'Medium sized',:]
df_Small = df1.loc[df1['company_category']== 'Small',:]
df_Very_large= df1.loc[df1['company_category']== 'Very large',:]

# fill the NAN values in each feature based on the mean values of the same category
years = ['2015', '2016','2017','2018','2019','2020']
Features = ['ebit', 'ebitda' , 'profit_and_loss_after_taxes' , 'total_assets' , 'total_liabilities' , 'operating_profit_and_loss' , 'financial_profit_and_loss' , 'staff_count', 'revenue' , 'net_added_value' , 'staff_costs']
for feature in Features:
    for year in years:
        df_Large[f'{feature}.{year}'][df_Large[f'{feature}.{year}'].isna()] = df_Large[f'{feature}.{year}'].mean()
        df_Medium[f'{feature}.{year}'][df_Medium[f'{feature}.{year}'].isna()] = df_Medium[f'{feature}.{year}'].mean()
        df_Small[f'{feature}.{year}'][df_Small[f'{feature}.{year}'].isna()] = df_Small[f'{feature}.{year}'].mean()
        df_Very_large[f'{feature}.{year}'][df_Very_large[f'{feature}.{year}'].isna()] = df_Very_large[f'{feature}.{year}'].mean()

    # concatenate the subsets to get the clean dataframe
Total_clean_df = pd.DataFrame()
all_df1 = [df_Large,df_Medium,df_Small, df_Very_large ]
for each_df in all_df1:
    Total_clean_df = pd.concat([Total_clean_df,each_df])
Total_clean_df.shape

Total_clean_df.to_csv('/Users/Jorg/Accounton data/clean_accounton.csv', index=False)
