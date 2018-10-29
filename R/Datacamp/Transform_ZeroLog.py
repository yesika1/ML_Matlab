
# coding: utf-8

# In[4]:


get_ipython().magic(u'pylab inline')


# In[5]:


import dataiku
from dataiku import pandasutils as pdu
import pandas as pd


# In[11]:


# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_stg_macro_econ_HDFS = dataiku.Dataset("stg_macro_econ_HDFS")
df = dataset_stg_macro_econ_HDFS.get_dataframe(limit=100000)


# In[ ]:


# Get some simple descriptive statistics
pdu.audit(df)


# In[82]:


x =np.log(0.0000000001)
x


# In[120]:


import numpy as np
def log_col2(col, colname='Column'):
    ''' function that calculates the log of the original variables
    '''
    value= np.log(np.where(col>0))
    colname = 'log_'+ colname
    return colname,value 
 


# In[101]:


import numpy as np
def log_col(col, colname='Column'):
    ''' function that calculates the log of the original variables

    '''
    for i in col:
        
        if i >0 or i==None: return 0
        else: 
            value= np.log(col)
            colname = 'log_'+colname
            return colname,value
        
#result = np.where(myarray>0, np.log(myarray), 0)        


# In[121]:


import numpy as np

def lag_difference(col,n=0,y=False,log=False, lag=False, colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    Where colname: QD= Quarterly Difference, OR  YD= Quarterly Difference
    log= True for calculations of the log of the difference
    lag= True for calculations of the lag of the difference

    '''
    # QD for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD'

    #elif y ==False and log==True and lag==False:
        #value= np.log(col.shift(n)) -np.log(col.shift(n+1))
        #colname = colname +'_logQD1'
        
    elif y ==False and log==True and lag==False:
        value= log_col2(col.shift(n))[1] -log_col2(col.shift(n+1))[1]
        colname = colname +'_logQD2'    
    
    return colname, value   


# In[95]:


import numpy as np

def lag_difference(col,n=0,y=False,log=False, lag=False, colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    Where colname: QD= Quarterly Difference, OR  YD= Quarterly Difference
    log= True for calculations of the log of the difference
    lag= True for calculations of the lag of the difference

    '''
    # QD for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD'

    elif y ==False and log==True and lag==False:
        value= np.log(col.shift(n)) -np.log(col.shift(n+1))
        colname = colname +'_logQD'

    elif y ==False and log==False and lag==True:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD' +'Lag'+str(n)

    # YD for Yearly calculations
    elif y ==True and log==False and lag==False:
        value= col.shift(n) - col.shift(n+4)
        colname = colname +'_YD'

    elif y ==True and log==True and lag==False:
        value= np.log(col.shift(n)) - np.log(col.shift(n+4))
        colname = colname +'_logYD'

    elif y ==True and log==False and lag==True:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_YD' +'Lag'+str(n)


    return colname, value


# In[122]:


import numpy as np
np.warnings.filterwarnings('ignore')

def transformations(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''
    data.sort_values(colSort,inplace=True) #sort values

    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):

            # Log transformation of original variables:
            colname, data[colname] = log_col2(data[c],colname= c)
            
            #log1
            colname, data[colname] = lag_difference(data[c],colname= c,y =False, log=True,lag=False)
            #log2
            colname, data[colname] = lag_difference(data[c],colname= c,y =False, log=True,lag=False)

    return data


# In[123]:


pd_transformed =transformations(df,'Date',n=4)


# In[77]:


pd_transformed.head(110)


# In[57]:


pd_transformed.columns


# In[32]:


df.head(20)

