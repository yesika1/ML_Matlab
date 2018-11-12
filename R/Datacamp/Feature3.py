
# coding: utf-8

# # Feature Importance G/B Model (PLOC)

# Model XGBoost (XGBoost), trained on 2018-11-08 09:01:20.

# This notebook will reproduce the steps for a BINARY_CLASSIFICATION on the CCAP_CAD_DAS_PERFORMANCE_MODELING table. The main objective is to highlight the feature importance

# Let's start with importing the required libs :

# In[1]:


get_ipython().magic(u'pylab inline')


# In[2]:


import dataiku
from operator import itemgetter
import math
from datetime import datetime
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from xgboost import plot_importance
from lightgbm import plot_importance
import lightgbm as lgb
from sklearn import linear_model
import xgboost as xgb
import matplotlib.pyplot as plt
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter


# And tune pandas display options:

# In[3]:


pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# #### Importing base data

# The first step is to get our machine learning dataset:

# In[4]:


# We apply the preparation.
ml_dataset_handle = dataiku.Dataset('CCAP_CAD_DAS_PERFORMANCE_MODELING')
get_ipython().magic(u'time ml_dataset = ml_dataset_handle.get_dataframe()')

print ('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)


# Back-up dataframe

# In[5]:


df = ml_dataset.copy()


# #### G/B data clustring
# Drop rows with null Target value

# In[6]:


def drop_na_rows_target(dataframe):
    dataframe = dataframe[dataframe['Target'].notnull()]
    return dataframe

out_of_sample = drop_na_rows_target(df)


# In[7]:


print ('Out-of-Sample base data has %i rows and %i columns' % (out_of_sample.shape[0], out_of_sample.shape[1]))
print ('Total number of customers who were rejected, or accepted and not booked', (df.shape[0] - out_of_sample.shape[0]))
# Five first records",
out_of_sample.head(5)


# **Retrive data following the TimeFrame - 2013/01/01 to 201/09/30**

# In[8]:


out_of_sample['OPENDATE'] = pd.to_datetime(out_of_sample['OPENDATE'] , format ='%Y-%m-%d').dt.date


# In[10]:


out_of_sample = out_of_sample.loc[(out_of_sample['OPENDATE'] >= datetime.date(2009, 1, 1)) 
                            & (out_of_sample['OPENDATE']  <= datetime.date(2011,9, 30))]


# In[11]:


print('Min date: ', out_of_sample['OPENDATE'].min())
print('Max date: ', out_of_sample['OPENDATE'].max())


# And drop empty columns where the values weren't recorded within out time frame

# In[12]:


out_of_sample = out_of_sample.dropna(how = 'all', axis = 1)


# Sample size

# In[13]:


len(out_of_sample)


# Distribution of our target variable

# In[14]:


out_of_sample['Target'].hist()


# Changing string format to date format

# In[15]:


def to_date(dataframe):
    
    dates = []
    time = []
    
    for column in dataframe.columns:
        
        if '_dt' in column or '_dtm' in column or '_dob' in column:
            print('Processing date format ......')
            #date
            #MBA_crte_dt feature has hours as well
            if '_dtm' in column:
                print('*', column)
                dataframe[column] = pd.to_datetime(dataframe[column] , format ='%d%b%Y:%H:%M:%S').dt.date
                dates.append(column)
            #Due to the hight volume of missing values, pandas turns dates into float instead of string. To be able
            # to change the format of those features into integer then date, several steps are needed.
            elif column == 'lastest_bkrp_dischrg_dt'            or column == 'lastest_bkrp_dt'             or column == 'lastest_clct_dt'or column == 'lastest_clct_dt'or column.startswith('top') or column.startswith('SF') or column.startswith('BI') or column.startswith('BR') or column.startswith('NC') or column.startswith('PF') or column.startswith('RT') or column.startswith('RD') or column.startswith('RE') or column.startswith('LC'):
                print('Processing hight volume of missing date format ......')
                print('*', column)
                dataframe[column] = dataframe[column].fillna(32199).astype(int)
                dataframe[column] = dataframe[column].astype('int64', copy=False)
                dataframe[column] = pd.to_datetime(dataframe[column] , format ='%m%Y').dt.date
                dataframe[column] = dataframe[column].replace(datetime.date(2199, 3, 1), np.nan)
                dates.append(column)
                
            else:
                print(column)
                dataframe[column] = pd.to_datetime(dataframe[column] , format ='%d%b%Y').dt.date
                dates.append(column)

            
        elif '_tm' in column:
            #Time
            print('Processing time format......')
            print('*', column)
            dataframe[column] = pd.to_datetime(dataframe[column] , format ='%H:%M:%S').dt.time
            
            time.append(column)
        
    print('Total features with a date format ', len(dates))
    print('Total features with a time format ', len(time))
    
    return dates, time, dataframe
    
li_dates, li_time, out_of_sample = to_date(out_of_sample)


# In[16]:


print ('Base out-of-sample data has %i rows and %i columns' % (out_of_sample.shape[0], out_of_sample.shape[1]))
# Five first records",
out_of_sample.head(5)


# #### Features preprocessing

# **Due to the high dimentionality of our dataset, we will try to manually delete categorical variables that don't necessary reflect observed behaviors or have typos.** (Kernel dies)

# Data cleaning is needed for the following features

# - pri_cty
# - pri_prev_cty
# - CCH_comment
# - CCH_comment_MECH

# In[17]:


#This function tries to change all data into float in case a feature holds a wrong Object data type. This step will minimize
#the number of feature in getting bigger while encoding.
def clear_dtype(dataframe):

    for col in dataframe.columns:
        try:
            dataframe[col] = dataframe[col].astype('float64', copy=False)
        except:
            pass

def drop_unecessary(dataframe):
    total_col = dataframe.shape[1]
    
    #From Performance table
    dataframe = dataframe.drop(['limits', 'BALANCE', 'actsysid', 'WROF_AMT', 'block', 'status', 'WROF_AMT', 'WROFDATE'], axis = 1)
    for col in dataframe.columns:
        
        if col == 'pri_cty' or col == 'pri_prev_cty' or col == 'CCH_comment' or col.endswith('_ref_no') or col.endswith('acct_nbr') or 'acct_num' in col or col.endswith('_id') or col == 'ocif':
            
            print('Colum respects the condition: ', col)
            dataframe.drop(col, axis = 1, inplace = True)
            
    print('Total dropped columns: ', total_col - dataframe.shape[1])
    
    return dataframe

out_of_sample = drop_unecessary(out_of_sample)


# In[18]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after drop ', len(out_of_sample.columns))


# Dates feature engineering
# > for every feature with a date format, the next step will be to split the day, month and year for every feature into three new features.

# In[19]:


def date_to_features(li_da, li_ti, dataframe):
    data = dataframe.copy()
    #date feature engineering
    for col in li_da:
        data[col + '_day'] = [d.day if type(d) == datetime.date else np.nan for d in data[col]]
        data[col + '_month'] = [m.month  if type(m) == datetime.date else np.nan for m in data[col]]
        data[col + '_year'] = [y.year if type(y) == datetime.date else np.nan for y in data[col]]
        
        del data[col]
     
    #time feature engineering
    for c in li_ti:
        data[c + '_hour'] = [h.hour if type(h) == datetime.date else np.nan for h in data[c]]
        data[c + '_minute'] = [m.minute if type(m) == datetime.date else np.nan for m in data[c]]
        data[c + '_second'] = [s.second  if type(s) == datetime.date else np.nan for s in data[c]]
        
        del data[c]
    
    #Opendate
    data['OPENDATE' + '_day'] = [i.day if type(i) == datetime.date else np.nan for i in data['OPENDATE']]
    data['OPENDATE' + '_month'] = [i.month if type(i) == datetime.date else np.nan for i in data['OPENDATE']]
    data['OPENDATE' + '_year'] = [i.year  if type(i) == datetime.date else np.nan for i in data['OPENDATE']]
    
    del data['OPENDATE']
    return data

out_of_sample = date_to_features(li_dates, li_time, out_of_sample)


# In[20]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after date engineering ', len(out_of_sample.columns))


# Drop empty and unique values columns

# In[21]:


def too_unique(dataframe):
    dataframe = dataframe.dropna(how='all', axis = 1)
    #Keep rows with non-unique values
    for col in dataframe:
        if(len(dataframe.loc[:,col].unique())  == 1):
            print('100% unique values',col, dataframe.loc[:,col].unique())
            dataframe.pop(col)
                
    return dataframe

out_of_sample = too_unique(out_of_sample)


# In[22]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after dropping empty columns ', out_of_sample.shape[0])
print('Total rows after dropping empty columns ', out_of_sample.shape[1])


# Label encoding for categorial variables

# In[ ]:


from sklearn.preprocessing import LabelEncoder

def label_encoding(dataframe):
    
    enc = LabelEncoder()
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object': 
            enc.fit(dataframe[col].astype(str))
            dataframe[col] = enc.transform(dataframe[col].astype(str))
                
    return dataframe
                
def hot_encoding(dataframe):
    
    return pd.get_dummies(dataframe, dummy_na = False )

out_of_sample = label_encoding(out_of_sample)
# out_of_sample = hot_encoding(out_of_sample)


# In[ ]:


print ('Encoded Out-of-Sample data has %i rows and %i columns' % (out_of_sample.shape[0], out_of_sample.shape[1]))
# Five first records",
out_of_sample.head(5)


# Before applying the data to our estimators, let's take a look a the movement of our target variable throught the years with respect to our time frame.

# ## Estimator

# In[ ]:


import xgboost as xgb
X = out_of_sample.drop('Target', axis = 1)
y = np.array(out_of_sample['Target'])


# **XGboost**

# In[ ]:


def XGB_feat_importance(X_train, y_train):
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, verbose=True)
    ax = plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 60)
    plt.show()
    
    return {'clf':clf}


get_ipython().magic(u'time XGB_feat_importance(X, y)')


# **LightXGboost**

# In[ ]:


def LXGB_feat_importance(X_train, y_train):
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train, verbose=True)
    ax = plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 60)
    plt.show()
    
    return {'clf':clf}


get_ipython().magic(u'time LXGB_feat_importance(X, y)')

