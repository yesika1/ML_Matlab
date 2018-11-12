
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
import sklearn as sk
from xgboost import plot_importance
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


# **Retrive data following the TimeFrame - 2009/01/01 to 2011/09/30**

# In[8]:


out_of_sample['OPENDATE'] = pd.to_datetime(out_of_sample['OPENDATE'] , format ='%Y-%m-%d').dt.date


# In[9]:


out_of_sample = out_of_sample.loc[(out_of_sample['OPENDATE'] >= datetime.date(2009, 1, 1)) 
                            & (out_of_sample['OPENDATE']  <= datetime.date(2011, 9, 30))]


# And drop empty columns where the values weren't recorded within out time frame

# In[10]:


out_of_sample = out_of_sample.dropna(how = 'all', axis = 1)


# Sample size

# In[11]:


len(out_of_sample)


# Distribution of our target variable

# In[12]:


out_of_sample['Target'].hist()


# Changing string format to date format

# In[13]:


def to_date(dataframe):
    
    dates = []
    time = []
    
    for column in dataframe.columns:
        
        if '_dt' in column or '_dtm' in column or '_dob' in column:
            #date
            #MBA_crte_dt feature has hours as well
            if column == 'MBA_run_dtm':
                print('Processing date format ......')
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
                print('Processing date format ......')
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


# In[14]:


print ('Base out-of-sample data has %i rows and %i columns' % (out_of_sample.shape[0], out_of_sample.shape[1]))
# Five first records",
out_of_sample.head(5)


# #### Features preprocessing

# **Due to the high dimentionality of our dataset, we will try to manually delete categorical variables that don't necessary reflect observed behaviors or have typos. Data cleaning is needed for the following features.** (Kernel dies)

# - pri_cty
# - pri_prev_cty
# - CCH_comment
# - CCH_comment_MECH'

# The following features have a high variance which leads to a sparse matrix and a high number of features. This method is not necessarily right. The only reason is to fit the data into the model using Pandas, otherwise, the kernel dies due to a low memory.

# - lia_card_desc_2
# - lia_card_desc_1
# - sec_cur_ocptn_desc
# - pri_cur_ocptn_desc
# - prev_brwr_br
# - appl_last_upd_user
# - adjud_user
# - prev_brwr_br
# - cls_user
# - book user
# - Every feature that ends with _trd_stat

# In[15]:


#This function tries to change all data into float in case a feature holds a wrong Object data type. This step will minimize
#the number of feature to get bigger while encoding.
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
        
        if col == 'lia_card_desc_2 ' or col == 'lia_card_desc_1' or col == 'sec_cur_ocptn_desc' or col == 'pri_cur_ocptn_desc' or col == 'pri_cty' or col == 'pri_prev_cty' or col == 'CCH_comment' or col == 'prev_brwr_br' or col == 'appl_last_upd_user' or col == 'adjud_user' or col == 'prev_brwr_br' or col == 'cls_user' or col == 'CCH_comment_MECH' or col == 'book_user' or col.endswith('_trd_stat') or col.endswith('_ref_no') or col.endswith('acct_nbr') or 'acct_num' in col or col.endswith('_id') or col == 'book' or col == 'ocif':
            
            print('Colum respects the condition: ', col)
            dataframe.drop(col, axis = 1, inplace = True)
            
    print('Total dropped columns: ', total_col - dataframe.shape[1])
    
    return dataframe

out_of_sample = drop_unecessary(out_of_sample)


# In[16]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after drop ', len(out_of_sample.columns))


# Dates feature engineering
# > for every feature with a date format, the next step will be to split the day, month and year for every feature into three new features.

# In[17]:


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


# In[18]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after date engineering ', len(out_of_sample.columns))


# Drop empty and unique values columns

# In[19]:


def too_unique(dataframe):
    dataframe = dataframe.dropna(how='all', axis = 1)
    #Keep rows with non-unique values
    for col in dataframe:
        if(len(dataframe.loc[:,col].unique())  == 1):
            print('100% unique values',col, dataframe.loc[:,col].unique())
            dataframe.pop(col)
                
    return dataframe

out_of_sample = too_unique(out_of_sample)


# In[20]:


print('Total columns at start ', len(ml_dataset.columns) )
print('Total columns after dropping empty columns ', len(out_of_sample.columns))


# Label encoding for categorial variables

# In[21]:


from sklearn import preprocessing    
def label_encoding(dataframe):
    enc = preprocessing.LabelEncoder()
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            if  len(list(dataframe[col].unique())) <= 3:
                enc.fit(dataframe[col].values.astype('str'))
                dataframe[col] = enc.transform(list(dataframe[col].values.astype('str')))

    return dataframe

out_of_sample = label_encoding(out_of_sample)


# In[22]:


def dummy(dataframe):
    
    return pd.get_dummies(dataframe, dummy_na= False)

out_of_sample = dummy(out_of_sample)


# In[23]:


print ('Out-of-Sample encoding data has %i rows and %i columns' % (out_of_sample.shape[0], out_of_sample.shape[1]))
# Five first records",
out_of_sample.head(5)


# ## Estimator

# In[24]:


import xgboost as xgb
X = out_of_sample.drop('Target', axis = 1)
y = np.array(out_of_sample['Target'])


# In[ ]:


def XGB_feat_importance(X_train, y_train):
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train, verbose=True)
    ax = plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 60)
    plt.show()
    
    return {'clf':clf}


get_ipython().magic(u'time XGB_feat_importance(X, y)')


# ## Correlation Analysis
# **Following Sheikh work**

# ## Correlation amongst continuous variables

# In[ ]:


Continous = out_of_sample[out_of_sample[col].dtype != 'object']
categorical = out_of_sample[out_of_sample[col].dtype == 'object']


# In[ ]:


f, ax = plt.subplots(figsize=(40, 40))
corr = Continous.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.title("Correlation Amongst Continuous Variables")


# ## Correlation against Target - Continuous

# In[ ]:


f, ax = plt.subplots(figsize=(5, 50))
sns.heatmap(pd.DataFrame([np.absolute(np.corrcoef(y, categorical[col])[1, 0]) for col in categorical.columns],
                         index=list(X.columns)), ax = ax)
plt.title("Absolute Correlation against Target - Continuous Variables")


# ## $\sqrt{R^{2}}$ against Target - Categorical Variables

# For categoricals, lets create a defined function which calculates the squard root of R Squared based on a linear model 

# In[ ]:


def abs_corr(data, y):
    
    abs_corr = []
    lm = linear_model.LinearRegression()

    for col in data.columns:
        x = pd.get_dummies(data[col]).values
        lm.fit(x, y)
        y_pred = lm.predict(x)
        abs_corr.append(np.sqrt(np.max([0,r2_score(y, y_pred)])))
    
    abs_corr = pd.DataFrame(abs_corr, index = list(data.columns))
    
    return abs_corr


# In[ ]:


f, ax = plt.subplots(figsize=(5, 50))
sns.heatmap(pd.DataFrame(abs_corr(categorical, categorical['Target']),
                         index= list(X_train_cat.columns)), ax = ax)
plt.title("Sqrt R2 - against Target - Categorical Variables")


# # Function for feature selections - ChiSquare and Mutual Information Score

# In[ ]:



def uni_variate_tests(X_train,y_train):
    
    print("Calculating chi_test statistics for all features...")
    chi_test = chi2(X_train[list(X_train.columns)], y_train)
    chi2_stat = chi_test[0]
    
    chi2_stat = pd.DataFrame(chi_test[0], index = list(X_train.columns)).    sort_values(by = 0, ascending = False)
    
    p_val = pd.DataFrame(chi_test[1], index = list(X_train.columns)).    sort_values(by = 0, ascending = True)
    
    print("Calculating mi_scores for all features...")
    
    mi_score = []
    for col in X_train.columns:
        mi_score.append(mutual_info_score(X_train[col].values, y_train))
        
    mi_score = pd.DataFrame(mi_score, index = list(X_train.columns)).sort_values(by = 0, ascending = False)          
    
    print('Done!')
    return {'chi2_stat': chi2_stat, 'p_val' : p_val, 'mi_score':mi_score}


# In[ ]:


uni_variate_analysis = uni_variate_tests(out_of_sample, out_of_sample['Target'])


# In[ ]:


chi_res = uni_variate_analysis['chi2_stat'].sort_values(by = 0, ascending = False).head(100)
chi_plt = chi_res.plot(kind = 'barh')
plt.gca().invert_yaxis()
chi_plt.figure.set_size_inches(10, 30)
chi_plt.set_title("Scores - Univariate Chi Square Test")


# Mutual Information Score:
# $$I(X;Y) = \sum_{y}\sum_{x}p(x, y)log(\frac{p(x,y)}{p(x)p(y)})$$

# In[ ]:


mi_score_head = uni_variate_analysis['mi_score'].sort_values(by = 0, ascending = False).head(100)
mis_plt = mi_score_head.plot(kind = 'barh')
plt.gca().invert_yaxis()
mis_plt.figure.set_size_inches(10, 30)
mis_plt.set_title("Mutual Information Score")

