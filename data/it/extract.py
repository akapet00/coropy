import pandas as pd 
import numpy as np 

df = pd.read_csv('raw/daily.csv', header=[0], thousands=',') # thousands w/out comma
headers = ['Date','New Confirmed', 'Total Confirmed', 'New Deaths', 'Total Deaths', 'Total Rec']
df = df[headers] # stats for entire Italy
df.replace('–', 0, inplace=True) # remove long - with 0
df['Total Rec'] = df['Total Rec'].str.replace(',', '') # remove comma
df.fillna(0, inplace=True) # remove nans if any for the whole df
df['Total Rec'] = df['Total Rec'].astype(int) # change the column type to int
print(df.info()) 
print(df)
data = [df['Total Confirmed'], df['Total Rec'], df['Total Deaths']]

files = ['confirmed_cases.dat', 'recovered_cases.dat', 'death_cases.dat']
for i, file in enumerate(files):
    with open(file, 'w') as f:
        for item in data[i]:
            f.write(f'{item}\n')
