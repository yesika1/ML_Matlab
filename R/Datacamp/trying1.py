
# coding: utf-8

# In[1]:


get_ipython().magic(u'pylab inline')


# In[2]:


# importing the libraries
import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from itertools import compress
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder)


# dataiku related libraries
import dataiku.core.pandasutils as pdu

# setting pandas options
pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# In[3]:


# Example: load a DSS dataset as a Pandas dataframe
ar = dataiku.Dataset("CCAP_CAD_DAS_DEV")
ar_df = ar.get_dataframe()

Let's sneakpeak the data 
# In[ ]:


ar_df.shape


# In[168]:


ar_df.head()


# In[169]:


pop_count = ar_df.groupby("accept")["accept"].count()
n = ar_df.shape[0]
pop_prcnt = pop_count/n

plt.bar(x = ["accept", "reject"] , height = pop_prcnt)
plt.xlabel("Class")
plt.ylabel("%")
plt.title("Distribution of accounts")


# Let's do a basic info rundown of the files 

# In[159]:


ar_df.info()


# Let's skim over the numerical variables and summarise them.

# In[160]:


ar_df.describe()


# Find the proportion of missing values for each column

# In[4]:


# This function will calculate the percentage of missing values across the dataset 
# and list the variables 
def find_missing(data):
    Total = data.isnull().sum().sort_values(ascending = False)
    Percentage = (data.isnull().sum()/data.shape[0]).sort_values(ascending = False)
    
    return pd.concat([pd.DataFrame(Percentage.index), pd.DataFrame(Total.values), 
                      pd.DataFrame(Percentage.values)],
                     axis = 1, keys = ['Var_names','Total','Percent'])


# In[5]:


# The function below 
def drop_miss(data, threshold):
    
    missing_stat = find_missing(data)
    varlist_threshold = list(missing_stat['Var_names']                          [missing_stat['Percent'] >= threshold].dropna()[0])
    new_df =  data.drop(labels= varlist_threshold, axis = 1)
    
    return {'new_df' : new_df, 'missing_stat' : missing_stat}

def drop_dup(data):
    
    return data.drop_duplicates()


# In[6]:


drop_miss_out = drop_miss(ar_df, 0.90)
ar_df = drop_miss_out['new_df']
ar_df.head()


# In[172]:


ar_df.shape


# In[7]:


ar_df = drop_dup(ar_df)


# In[174]:


ar_df.shape


# In[175]:


ar_df.describe()


# In[8]:


def drop_redundant_vars(data):
    
    var_names = list(data.columns)
    for col in var_names:

        if col.endswith('_ref_no')        or col.endswith('acct_nbr')        or col.endswith('_id')        or col.endswith('_cd')        or col.startswith('_cls')        or 'book' in col        or col == 'ocif': 

            data = data.drop(col, axis = 1)

    print("Total number of columns taken out ", len(var_names) - data.shape[1])

    return data


# In[9]:


ar_df = drop_redundant_vars(ar_df)


# In[31]:


# need to deal with the date type columns, i believe they end with _dt
def ID_date_vars(data):
    dt_tag = []
    var_names = list(data.columns)
    for i in range(len(var_names)):
        var_i = var_names[i].endswith('_dt')
        dt_tag.append(var_i)

    date_vars = list(compress(var_names, dt_tag))
    return data[date_vars]


# In[33]:


date_var_df = ID_date_vars(ar_df)
date_var_df.head()


# In[ ]:


# converting date columns to integers

strp_dt = datetime.datetime.strptime(dt, '%d%b%Y')
if strp_dt.day in range(9):
    

datetime.datetime.strptime('20APR2009', '%d%b%Y')


# In[195]:


num_var_ids = list(ar_df.select_dtypes(include= [np.number]).columns)
cat_var_ids = list(ar_df.select_dtypes(include= [object]).columns)


# In[196]:


def too_nunique_drop(data, cat_var_ids, nunique_thresh = 100):
    n_unique = data[cat_var_ids].nunique()
    varlist_threshold = list(n_unique[n_unique > 100].index)
    new_df =  data.drop(labels= varlist_threshold, axis = 1)
    
    return {'new_df' : new_df, 'varlist_threshold' : varlist_threshold}


# In[197]:


ar_df = too_nunique_drop(data = ar_df, cat_var_ids = cat_var_ids)['new_df']


# In[198]:


shape(ar_df)


# In[53]:


t = datetime.datetime.strptime('20APR2009', '%d%b%Y')


# In[60]:


int(str(t.day)+str(t.month) +str(t.year))


# In[64]:


range(9)

