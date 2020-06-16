# -*- coding: utf-8 -*-
"""bank-project-lifecoding-20200530.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZglqL0SO-LVWkDDaoncMaHXYiBywSlGb
"""

import pandas as pd

!ls 'drive/My Drive/data'

df = pd.read_csv('drive/My Drive/data/credit_scoring_eng.csv')

df.head()

len(df)

"""# Identify and fill in missing values."""

df.isna().mean()

df['days_employed'].median()

df['days_employed'].fillna(df['days_employed'].median())

"""# Replace the real number data type with the integer type"""

df.dtypes

df

import numpy as np

pd.isnull(np.nan)

pd.isnull(1)

def get_int(x):
  if pd.isnull(x):
    return x
  return pdint(x)

get_int(np.nan)

df['days_employed'].apply(get_int).dtype

type(np.nan)

-99999999

df['days_employed'].min()

df['days_employed_int'] = df['days_employed'].fillna(-99999999).round().astype(int)

df

t = pd.DataFrame({'x':[1.2,1.6]})

t

t['x'].astype(int)

df

"""# Delete duplicate data"""

df.duplicated().mean()

df.duplicated().sum()

df[df.duplicated()]

df['education'] = df['education'].str.lower()

for col in ['family_status','income_type','purpose']:
  df[col] = df[col].str.lower()

df

df.duplicated().sum()

df[df.duplicated()].isna().mean()

"""# Categorize the data"""

df.columns

len(df['purpose'].value_counts())

df['purpose'].value_counts()

def get_cat(x):
  if 'wedding' in x:
    return 'wedding'
  if 'real estate' in x or 'hous' in x or 'proper' in x:
    return 'real estate'
  if 'car' in x:
    return 'car'
  if 'educa' in x or 'university' in x :
    return 'educat'
  return 'misc'

df['purpose_group'] = df['purpose'].apply(get_cat)

df['purpose_group'].value_counts(normalize=True)

df['purpose_group'].value_counts()

df[df['purpose_group'] == 'misc']['purpose'].value_counts()

"""# Is there a connection between having kids and repaying a loan on time"""

pd.concat([df['children'].value_counts(),
           df['children'].value_counts(normalize=True)],axis=1)

t = df.groupby('children')['debt'].agg(['count','mean']).reset_index()

t[t['count'] > 2000]

df['has_kids'] = (df['children'] > 0) * 1

df.groupby('has_kids')['debt'].mean()

"""# Is there a connection between marital status and repaying a loan on time?"""

df.groupby('family_status')['debt'].agg(['count','mean']).reset_index()

pd.concat([df['family_status'].value_counts(),
           df['family_status'].value_counts(normalize=True)],axis=1)

"""# Is there a connection between income level and repaying a loan on time?"""

df['total_income'].median()

df['total_income_fixed'] = df['total_income'].fillna(-999999)

pd.cut(df['total_income_fixed'],[-np.inf,0,20000,30000,50000,np.inf]).value_counts()

df['income_group'] = pd.cut(df['total_income_fixed'],[-np.inf,0,20000,30000,50000,np.inf])

df.groupby('income_group')['debt'].agg(['count','mean']).reset_index()

df.groupby('income_group')['debt'].mean().plot(grid=True,ylim=0)

"""ncome_level
High         0.069697
Low          0.083349
Medium       0.080071
No Income    0.078197

0.005
"""

df

pd.concat([df['gender'].value_counts(),
           df['gender'].value_counts(normalize=True)],axis=1)

pd.concat([df['income_type'].value_counts(),
           df['income_type'].value_counts(normalize=True)],axis=1)

"""# How do different loan purposes affect timely loan repayment?"""

df.groupby('purpose_group')['debt'].agg(['count','mean']).reset_index()

t = pd.DataFrame({'user_id':[1,2,3],'dur':[2,3,2],'tariff':['a','b','b']})

t

tt = pd.DataFrame({'tariff':['a','b'],'dur_cost':[20,40]})

tt

t = t.merge(tt,how='left',on='tariff')

t['revenue'] = (t['dur'] * t['dur_cost']) + (t['mb'] * t['mb_cost'])

t
