
# coding: utf-8

# ##  Recipe for the Transformation of Macroeconomic Variables

# In[ ]:


get_ipython().magic(u'pylab inline')


# In[ ]:


import dataiku
from dataiku import pandasutils as pdu
import pandas as pd


# In[ ]:


# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_stg_macro_econ_HDFS = dataiku.Dataset("stg_macro_econ_HDFS")
df = dataset_stg_macro_econ_HDFS.get_dataframe(limit=100000)


# In[ ]:


# Get some simple descriptive statistics
pdu.audit(df)


# ## Macroeconomic Variables Transformation

# ### Log calculation per column

# In[ ]:


import numpy as np
def log_col(col, colname='Column'):
    ''' function that calculates the log of the original variables
        if value is 0, replace result for NaN
    '''
    value= np.log(col.replace(0, np.nan))
    colname = 'log_'+colname
    return colname,value

# ### Transformation for Quarterly and Yearly difference.
# 
# Definition of the function that calculates the Variable difference: Q1 = X(t)  - X(t-1)
# * Where column names:
#     * QD= Quarterly Difference,
#     * YD= Yearly Difference

# In[ ]:


import numpy as np

def lag_difference(col,n=0,y=False,log=False, lag=False, colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    Where colname: QD= Quarterly Difference, OR  YD= Quarterly Difference
    log= True for calculations of the log of the difference
    lag= True for calculations of the lag of the difference
    '''
    # for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD'

    elif y ==False and log==True and lag==False:
        value= log_col(col.shift(n))[1] -log_col(col.shift(n+1))[1]
        colname = colname +'_logQD'

    elif y ==False and log==False and lag==True:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD' +'Lag'+str(n)

    # for Yearly calculations
    elif y ==True and log==False and lag==False:
        value= col.shift(n) - col.shift(n+4)
        colname = colname +'_YD'

    elif y ==True and log==True and lag==False:
        value= log_col(col.shift(n))[1] - log_col(col.shift(n+4))[1]
        colname = colname +'_logYD'

    elif y ==True and log==False and lag==True:
        value= col.shift(n) -col.shift(n+4)
        colname = colname +'_YD' +'Lag'+str(n)


    return colname, value


# ### Transformation for Quarterly and Yearly Growth.
# 
# Definition of the function that calculates the Variable difference: Q1 = X(t)  - X(t-1)
# * Where column names:
#     * QG= Quarterly Growth,
#     * YG= Yearly Growth

# In[ ]:


def lag_growth(col, n=0, y=False, log=False,lag=False, colname='Column'):
    ''' function that calculates the lag n lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100 and returns an array
    col = array, n = interger,number of lags calculations. 
    y:True(4 Quarters) if it is a YY calculation
    colname: string column name
    Where colname: QG= Quarterly Growth, OR  YG= Quarterly Growth
    log= True for calculations of the log of the difference
    lag =True for calculations of the lag of the difference
    '''
    # for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= ((col.shift(n) /col.shift(n+1))-1)*100
        colname = colname +'_QG'

    elif y ==False and log==True and lag==False:
        value= (log_col(col.shift(n))[1] / log_col(col.shift(n+1))[1] )-1
        colname = colname +'_logQG'

    elif y ==False and log==False and lag==True:
        value= ((col.shift(n) /col.shift(n+1))-1)*100
        colname = colname +'_QG' +'Lag'+str(n)

    # for Yearly calculations
    elif y ==True and log==False and lag==False:
        value= ((col.shift(n) /col.shift(n+4))-1)*100
        colname = colname +'_YG'

    elif y ==True and log==True and lag==False:
        value= (log_col(col.shift(n))[1] /log_col(col.shift(n+4))[1])-1
        colname = colname +'_YG'

    elif y ==True and log==False and lag==True:
        value= ((col.shift(n) /col.shift(n+4))-1)*100
        colname = colname +'_YG' +'Lag'+str(n)

    return colname, value


# ### Transformation applied to the Pandas dataframe
# 
# * Applying the difference and Growth calculations within quarters and years.
# * Applying the lag(1 to 4) of the previous calculations
# * Calling the functions: lag_difference & lag_growth.
# * Applying the log of the original variables

# In[ ]:


import numpy as np
np.warnings.filterwarnings('ignore')

def transformations(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    n is the number of lags to be applied to the columns
    '''
    data.sort_values(colSort,inplace=True) #sort values

    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):

            # Log transformation of original variables:
            colname, data[colname] = log_col(data[c],colname= c)

            # Quarter/Quarter difference: current quarter-last quarter
            colname, data[colname] = lag_difference(data[c],colname= c)

            # Quarter/Quarter growth: (current quarter-last quarter)-1
            colname, data[colname] = lag_growth(data[c],colname= c)

            # Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=False)

            # Year/year growth: (current year-last year)-1
            colname, data[colname] = lag_growth(data[c],colname= c, y=True)

            # log of Quarter/Quarter difference
            colname, data[colname] = lag_difference(data[c],colname= c, y=False, log=True)

            # log of Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=True)

            # LAGS:
            # Applying 1 to 4 lag transformation
            for i in range(1,n+1):
                # Applying 1 to 4 lag transformation
                # for Quarter diff transformations:
                colname, data[colname] = lag_difference(data[c],colname= c, n=i,lag=True)

                # for yearly diff transformations
                colname, data[colname] = lag_difference(data[c],colname= c, n=i,lag=True, y=True)

                # for Quarter growth transformations:
                colname, data[colname] = lag_growth(data[c],colname= c, n=i,lag=True)

                # for yearly growth transformations
                colname, data[colname] = lag_growth(data[c],colname= c, n=i,lag=True, y=True)
                
                # for log of Quarter/Quarter difference
                colname, data[colname] = lag_difference(data[c],colname= c, y=False, log=True,lag=True)

                # for log of Year/year difference: (current year-last year)
                colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=True,lag=True)
                

    return data


# In[ ]:


pd_transformed =transformations(df,'Date',n=4)


# In[ ]:


pd_transformed.head()


# In[ ]:


pd_transformed.info()


# ### Recipe outputs

# In[ ]:


# Recipe outputs
stg_macro_econ_transform2 = dataiku.Dataset("stg_macro_econ_transform2")
#stg_macro_econ_transform2.write_with_schema(pandas_dataframe)
stg_macro_econ_transform2.write_with_schema(pd_transformed)

