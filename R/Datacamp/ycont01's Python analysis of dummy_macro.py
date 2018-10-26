
# coding: utf-8

# In[1]:


get_ipython().magic(u'pylab inline')


# In[2]:


import dataiku
from dataiku import pandasutils as pdu
import pandas as pd


# In[33]:


# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_dummy_macro_1 = dataiku.Dataset("dummy_macro_1")
df = dataset_dummy_macro_1.get_dataframe(limit=100000)


# In[34]:


# Get some simple descriptive statistics
pdu.audit(df)


# In[35]:


import numpy as np
np.warnings.filterwarnings('ignore')
    
def transformations(data,colSort):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''
    
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):
            
            # Quarter/Quarter difference: current quarter-last quarter
            name_qd = str(c)+'_qd'
            data[name_qd] = data[c]- data[c].shift()
            
            # Quarter/Quarter growth: (current quarter-last quarter)-1
            name_qq = str(c)+'_qq'
            data[name_qq] = ((data[c] /data[c].shift()) -1)*100
            
            # Year/year difference: (current year-last year)
            name_yd = str(c)+'_yd'
            data[name_yd] = data[c]-(data[c].shift(periods=4))
           
            # Year/year growth: (current year-last year)-1
            name_yy = str(c)+'_yy'
            data[name_yy] = ((data[c]/data[c].shift(periods=4)) -1 )*100
            
            
            # log of QQ difference
            name_logqd = str(c)+'_logqd'
            data[name_logqd] = np.log( data[c]) - np.log(data[c].shift() )
            
            # log of Quarter/Quarter growth: (current quarter-last quarter)-1
            name_logqq = str(c)+'_logqq'
            data[name_logqq] = ( np.log(data[c])  / np.log(data[c].shift())-1 )
            
            # log of Year/year difference: (current year-last year)-1
            name_logyd = str(c)+'_logyd'
            data[name_logyd] = (np.log(data[c]) - np.log(data[c].shift(periods=4)) )
           
            # log of Year/year growth: (current year-last year)-1
            name_logyy = str(c)+'_logyy'
            data[name_logyy] = (np.log( (data[c])  / np.log(data[c].shift(periods=4))) -1  )
            # negative values inside the log, which gives nan with real numbers
            
            
            # Lag1: Quarter/Quarter difference: current quarter-last quarter
            name_lagqd = str(c)+'_lagqd'
            data[name_lagqd] = data[c].shift(1)- data[c].shift(2)
            
            # Lag1: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lagqq = str(c)+'_lagqq'
            data[name_lagqq] = ((data[c].shift(1) /data[c].shift(2)) -1)*100
            
            # Lag1: Year/year difference: (current year-last year)
            name_lagyd = str(c)+'_lagyd'
            data[name_lagyd] = data[c].shift(4)-(data[c].shift(periods=8))
           
            # Lag1: Year/year growth: (current year-last year)-1
            name_lagyy = str(c)+'_lagyy'
            data[name_lagyy] = ((data[c].shift(4)/data[c].shift(periods=8)) -1 )*100
 

            # Lag2: Quarter/Quarter difference: current quarter-last quarter
            name_lag2qd = str(c)+'_lag2qd'
            data[name_lag2qd] = data[c].shift(2)- data[c].shift(3)
            
            # Lag2: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag2qq = str(c)+'_lag2qq'
            data[name_lag2qq] = ((data[c].shift(2) /data[c].shift(3)) -1)*100
            
            # Lag2: Year/year difference: (current year-last year)
            name_lag2yd = str(c)+'_lag2yd'
            data[name_lag2yd] = data[c].shift(8)-(data[c].shift(periods=12))
           
            # Lag2: Year/year growth: (current year-last year)-1
            name_lag2yy = str(c)+'_lag2yy'
            data[name_lag2yy] = ((data[c].shift(8)/data[c].shift(periods=12)) -1 )*100
 

            # Lag3: Quarter/Quarter difference: current quarter-last quarter
            name_lag3qd = str(c)+'_lag3qd'
            data[name_lag3qd] = data[c].shift(3)- data[c].shift(4)
            
            # Lag3: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag3qq = str(c)+'_lag3qq'
            data[name_lag3qq] = ((data[c].shift(3) /data[c].shift(4)) -1)*100
            
            # Lag3: Year/year difference: (current year-last year)
            name_lag3yd = str(c)+'_lag3yd'
            data[name_lag3yd] = data[c].shift(12)-(data[c].shift(periods=16))
           
            # Lag3: Year/year growth: (current year-last year)-1
            name_lag3yy = str(c)+'_lag3yy'
            data[name_lag3yy] = ((data[c].shift(12)/data[c].shift(periods=16)) -1 )*100
 

            # Lag4: Quarter/Quarter difference: current quarter-last quarter
            name_lag4qd = str(c)+'_lag4qd'
            data[name_lag4qd] = data[c].shift(4)- data[c].shift(5)
            
            # Lag4: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag4qq = str(c)+'_lag4qq'
            data[name_lag4qq] = ((data[c].shift(4) /data[c].shift(5)) -1)*100
            
            # Lag4: Year/year difference: (current year-last year)
            name_lag4yd = str(c)+'_lag4yd'
            data[name_lag4yd] = data[c].shift(16)-(data[c].shift(periods=20))
           
            # Lag4: Year/year growth: (current year-last year)-1
            name_lag4yy = str(c)+'_lag4yy'
            data[name_lag4yy] = ((data[c].shift(16)/data[c].shift(periods=20)) -1 )*100    
    
    return data


# In[36]:


df =transformations(df,'Date')


# In[37]:


df.head(12)


# In[ ]:


def lag_dif(col,n=0,q=0,colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    colname: string column name
    '''
    if n == 0:
        if q ==0:
            return col - col.shift(n+1)
        else: return col - col.shift(n+4)
    else:
        if q ==0:return col.shift(n) -col.shift(n+1)
        else: return col.shift(n) - col.shift(n+4)   
        


# In[ ]:


def lag_growth(col,n=0,q=0):
    ''' function that calculates the lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    '''
    if n == 0:
        if q ==0: return ((col /col.shift(n+1))-1)*100        
        else: return ((col /col.shift(n+4))-1)*100
        
    else:
        return  ((col.shift(n) /col.shift(n+1))-1)*100


# In[126]:


def lag_dif(col,n=0,q=0,colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    colname: string column name
    '''
    if n == 0:
        if q ==0:
            value = col - col.shift(n+1)
            colname = colname +'_QD'
             
        else: 
            value= col - col.shift(n+4)
            colname = colname +'_YD'
    else:
        if q ==0:
            value= col.shift(n) -col.shift(n+1)
            colname = colname +'_QD'
        else: 
            value= col.shift(n) - col.shift(n+4)
            colname = colname +'_QD'
            
    return value, colname


# In[ ]:


def lag_growth(col,n=0,q=0):
    ''' function that calculates the lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    '''
    if n == 0:
        if q ==0: 
            value = ((col /col.shift(n+1))-1)*100  
            colname = colname +'_QG'
        else: 
            value= ((col /col.shift(n+4))-1)*100
            colname = colname +'_YG'
        
    else:
        if q ==0:
            value = ((col.shift(n) /col.shift(n+1))-1)*100
            colname = colname +'_QG'
        else:
            value= ((col.shift(n) /col.shift(n+4))-1)*100
            colname = colname +'_YG'
    


# In[ ]:


def name(colname):
      name_qd = str(c)+'_qd'  


# In[81]:


import numpy as np
np.warnings.filterwarnings('ignore')
    
def transformations(data,colSort,n=0):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):
            
            # Quarter/Quarter difference: current quarter-last quarter
            name_qd = str(c)+'_qd'
            data[name_qd] = lag_dif(data[c],0)
            
            #lags Quarter/Quarter difference:
            for i in range(1, n+1): 
            #lags Quarter/Quarter difference
                name_lagqd = str(c) +'_qd' +'_lag'+str(i)
                data[name_lagqd] = lag_dif(data[c],i)
                
                
            # Quarter/Quarter growth: (current quarter-last quarter)-1
            name_qq = str(c)+'_qq'
            data[name_qq] = lag_growth(data[c],0)
             
            #lags Quarter/Quarter growth:    
            for i in range(1, n+1): 
            #lags Quarter/Quarter difference
                name_lagqd = str(c) +'_qq' +'_lag'+str(i)
                data[name_lagqd] = lag_growth(data[c],i)               
                   
            
            # Year/year difference: (current year-last year)
            name_yd = str(c)+'_yd'
            data[name_yd] = data[c]-(data[c].shift(periods=4))
            
                for i in range(n+1): 
                #lags Year/year difference
                name_lagqd = str(c) +'_qd' +'_lag'+str(i+1)
                data[name_lagqd] = lag_n(data[c],i)
            
            
            
           
            # Year/year growth: (current year-last year)-1
            name_yy = str(c)+'_yy'
            data[name_yy] = ((data[c]/data[c].shift(periods=4)) -1 )*100
            
            
            # log of QQ difference
            name_logqd = str(c)+'_logqd'
            data[name_logqd] = np.log( data[c]) - np.log(data[c].shift() )
            
            # log of Quarter/Quarter growth: (current quarter-last quarter)-1
            name_logqq = str(c)+'_logqq'
            data[name_logqq] = ( np.log(data[c])  / np.log(data[c].shift())-1 )
            
            # log of Year/year difference: (current year-last year)-1
            name_logyd = str(c)+'_logyd'
            data[name_logyd] = (np.log(data[c]) - np.log(data[c].shift(periods=4)) )
           
            # log of Year/year growth: (current year-last year)-1
            name_logyy = str(c)+'_logyy'
            data[name_logyy] = (np.log( (data[c])  / np.log(data[c].shift(periods=4))) -1  )
            # negative values inside the log, which gives nan with real numbers
 

    return data


# In[ ]:


##function working, but output differenct in lag.not right


# In[119]:


## lag quarter diff
def lag_dif(col,n):
    ''' function that calculates the lag n difference: Q1 = X(t-1)  - X(t-2)
    '''
    if n == 0:
        return col -col.shift(n+1)
    else:
        return col.shift(n) -col.shift(n+1)
    


# In[120]:


## lags transformation
def lag_transformations(data,colSort, n):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''    
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("qd" in c ) | ("qq" in c ) | ("yd" in c ) | ("yy" in c )):
            
            for i in range(1,n+1): 
                # Applying 1 to 4 lag transformation
                # for Quarter transformations:
                name_lagqd = str(c)+'_lag'+str(i)
                data[name_lagqd] = lag_n(data[c],i)
   
    return data


# In[121]:


dataset_dummy_macro_1 = dataiku.Dataset("dummy_macro_1")
df = dataset_dummy_macro_1.get_dataframe(limit=100000)


# In[122]:


df_transformed = transformations(df,'Date')
# df_transformed.columns


# In[123]:


df_lag = lag_transformations(df_transformed,'Date', 4)


# In[124]:


#lagdf = lag_transformations(df_transformed,'Date')

for c in df_lag.columns: 
    #if (df_transformed[c].dtype != object) & ( ("qd" in c ) | ("qq" in c )):
    print(c)


# In[125]:


df_lag.head(50)

