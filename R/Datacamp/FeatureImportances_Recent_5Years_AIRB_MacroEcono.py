
# coding: utf-8

# In[99]:


##===========================================================
### Libraries
##===========================================================
get_ipython().magic(u'pylab inline')
import dataiku
import dataiku.spark as dkuspark
import pyspark
from pyspark.sql import SQLContext

import matplotlib.pyplot as plt

from pyspark.sql.types import StringType # to return strings in functions
from pyspark.sql.functions import udf #for user defined functions
from pyspark.ml.feature import StringIndexer # to transform categorical columns to label
import pyspark.sql.functions as func


import pandas as pd #Pandas data frame is prettier than Spark DataFrame.show().

# Load PySpark
conf = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), 
                                   ('spark.executor.instances','10')])
#80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


# ### load the AIRB_TIMEKEY_Date dataset

# In[100]:


# Example: Read the descriptor of a Dataiku dataset
mydataset = dataiku.Dataset("AIRB_TIMEKEY_Date")
# And read it as a Spark dataframe
df = dkuspark.get_dataframe(sqlContext, mydataset)


# ### load the Macro Economic Factors dataset

# In[101]:


# Example: Read the descriptor of a Dataiku dataset
mydataset2 = dataiku.Dataset("stg_macro_econ_transform2")
# And read it as a Spark dataframe
df_macro = dkuspark.get_dataframe(sqlContext, mydataset2)


# In[102]:


# the number of columns in the dataframe
len(df.columns)


# In[103]:


# the number of columns in the Macroeconomic factors
len(df_macro.columns)


# In[104]:


# select a few macroeconomic factors from the whole dataset
df_macro_few = df_macro.select("Date", "CA_UNEMPLOYMENT_RATE", "CA_CARealGDP", "CA_REALGDP_YY", "CA_REALGDP_QA", 
                               "CA_REALGDP_RE", "CA_HOUSEPRICE_YY", "CA_HOUSEPRICE_QA", "CA_AVG_BENCHMARK_CAP_RATE")

len(df_macro_few.columns)


# In[105]:


df_macro_few.count()


# In[20]:


df_macro_few.printSchema()


# In[4]:


# get a list of the column names for this dataframe
df.columns


# In[ ]:


# Examples: to count the total records in the dataframe
rows = df.count()
print(rows)


# In[5]:


# Show the content of the "TIME_DIM_KEY" column
# df.select('TIME_DIM_KEY').show()

# Show the distinct "TIME_DIM_KEY" in the dataframe
df.select('TIME_DIM_KEY').distinct().show()


# In[12]:


# Count how many distinct "TIME_DIM_KEY" in the dataframe
df.select('TIME_DIM_KEY').distinct().count()


# In[ ]:


# The time_dim_key is a key for a day period, then the period start date = period end date = calendar_date of the time_dim_key


# In[13]:


# Group by "TIME_DIM_KEY", count the records in the groups
df.groupBy('TIME_DIM_KEY').count().show()


# In[14]:


# Group by the target variable "y", count the records in the groups
df.groupBy('y').count().show()


# In[16]:


# count the distinct "STATUS"
df.select('STATUS').distinct().count()


# In[17]:


# Group by "STATUS", count the records in the groups
df.groupBy('STATUS').count().show()


# ## Choose the different **TIME_DIM_KEY** snapshots of the data for analysis and training. 

# In[106]:


from pyspark.sql.functions import col

# 20 snapshots (31/07/2012 to 30/04/2017)
# snapshot_list = ['37580', '37583', '37586', '37589', '37592', '37595', '37598', '37601', '37604', '37607', '37610', 
               #  '37613', '37616', '37619', '37622', '37625', '37628', '37631', '37634', '37637']

# 7 most recent snapshots 
snapshot_list = ['37619', '37622', '37625', '37628', '37631', '37634', '37637']

# Choose/filter those snapshots and return a filtered dataframe
df_snapshots = df.where(col('TIME_DIM_KEY').isin(snapshot_list))

# Whole data 10 years data
#df_snapshots = df

# Total records in the filtered dataframe
df_snapshots.count()


# The most recent 5 years data consisting of 20 snapshots is about 18.5 million rows; 
# the data size is still too large for Pandas dataframe

# In[107]:


get_ipython().run_cell_magic(u'time', u'', u'# df_snapshots.toPandas()  \n# failed. \n# Error message: Total size of serialized results of 246 tasks (2.0 GB) is bigger than spark.driver.maxResultSize (2.0 GB)')


# In[20]:


# Group by the target variable "y", count the records in the groups
df_snapshots.groupBy('y').count().show()


# The ratio between the Non-defaults : Defaults is about 200: 1, for the 7 snapshots of the data. 

# In[25]:


# drop duplicate rows from the df_snapshots
df_snapshots.dropDuplicates().count()


# There are no duplicate rows, as the total number of remaining rows keeps the same: 5290482 rows. 

# In[17]:


# Example: View the first 3 rows of the dataframe
df_snapshots.take(3)


# In[18]:


# Print the Schema of the dataframe
df_snapshots.printSchema()


# ## Missing value percentage for each variable

# In[ ]:


from pyspark.sql.functions import col, sum

def missing_values(dataframe):
    '''Display percentage of missing values within each feature
    Arg: spark dataframe
    return: pandas dataframe
    '''
    rows = dataframe.count() # count total rows of the dataframe
    
    # Calculate the total number of missing value for each column variable, then convert to Pandas dataframe, then transpose. 
    mis_val = dataframe.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataframe.columns)).toPandas().transpose()
    
    # insert one column on the right side to show the missing percentages
    mis_val.insert(1, 'Percentage of missing value', 100 * mis_val[0] / rows)
    
    # sort the table by the missing percentage descending
    mis_val.sort_values(['Percentage of missing value'], ascending=[False], inplace=True)
    
    # rename the first columns as "count" of missing
    mis_val = mis_val.rename(columns={0:'count'})
    
    # insert one column on the left side to show the labels of each variable.
    mis_val.insert(0, 'Variable label', mis_val.index.values)
    
    # return a Pandas table summarizing the missing percentages for each variable.
    return mis_val


# This function will return with a Pandas dataframe showing the missing value percentage for each variables in the input Pyspark dataframe. Since this missing value table is just a small size (436 rows * 2 columns), there is no harm to the space. 

# In[ ]:


missing_table = missing_values(df_snapshots)
# return a table summarizing the missing percentages for each variables

missing_table


# ## Take a random sample out of the **df_snapshots** dataframe 

# In[108]:


# Draw a 2% randomly sample rows from the base Dataframe
df_sample = df_snapshots.sample(False, fraction = 0.05, seed=42)


# In[109]:


df_sample.count()


# This is about 264781/5290482 = 5% of the 7 snapshots of data.

# In[43]:


# Check the "Date" and "TIME_DIM_KEY" are mapped correctly. 
df_sample.select('TIME_DIM_KEY', 'Date').distinct().show()


# In[110]:


# Verify total snapshot is 20. 
df_sample.select('TIME_DIM_KEY', 'Date').distinct().count()


# So we verified that 20 snapshots are selected into our sample data. 

# ### use the "Date" column to map the Macroeconomic factors

# In[111]:


# Merge (outer join) two dataframes based on the 'Date' column value. 
df_sample = df_sample.join(df_macro_few, on=['Date'], how='left_outer')


# In[112]:


len(df_sample.columns)


# In[66]:


df_sample.select('ACTSYSID', 'TIME_DIM_KEY', 'Date', "CA_UNEMPLOYMENT_RATE", "CA_CARealGDP", "CA_REALGDP_YY", "CA_REALGDP_QA", 
                               "CA_REALGDP_RE", "CA_HOUSEPRICE_YY", "CA_HOUSEPRICE_QA", "CA_AVG_BENCHMARK_CAP_RATE").show()


# In[113]:


# change the type of the element in the dataframe from string into double. 
df_sample = df_sample.select(*[col(c).cast("double") for c in df_sample.columns])

# Drop Channel, Status and Role as they are character type
df_sample = df_sample.drop('COUNT', 'Channel','Status', 'Role', 'TIME_DIM_KEY', 'Date', 'SP_ACT_KEY', 
                           'pdclass', 'PD_SCR_VAL', 'APPL_DIM_KEY')


# ### Convert this sample Pyspark dataframe into a Pandas dataframe

# In[114]:


# convert to Pandas dataframe
df_sample_pd =df_sample.toPandas()

# show the shape of this Pandas dataframe
df_sample_pd.shape


# In[115]:


df_sample_pd.head(20)


# In[116]:


# check the type of the dataframe

print (df_sample_pd.dtypes)


# ### Label encoder for the categorical variables

# In[117]:


# get the frequency counts of the data type of the elements in a variable over the entire Pandas dataframe.
dtypeCount = [df_sample_pd.iloc[:, i].apply(type).value_counts() for i in range (df_sample_pd.shape[1])]


# In[118]:


dtypeCount


# In[119]:


import numpy as np
from sklearn import preprocessing

categorical_feats = [f for f in df_sample_pd.columns if df_sample_pd[f].dtype == 'object']
print (len(categorical_feats))  

lb = preprocessing.LabelEncoder()
for colu in categorical_feats:
    lb.fit(list(df_sample_pd[colu].values.astype('str')))
    df_sample_pd[colu] = lb.transform(list(df_sample_pd[colu].values.astype('str')))

  


# In[120]:


df_sample_pd


# In[121]:


df_sample_pd.shape


# In[122]:


length = df_sample_pd.shape[1]
print (length)


# ### use XGBoost to rank the feature importances for all variables

# In[123]:


from xgboost import XGBClassifier
# from xgboost import gbtree
model = XGBClassifier()
X = df_sample_pd.iloc[:, 2:(length-1)]
y = df_sample_pd.iloc[:, 1]
mod = model.fit(X, y) 


# In[124]:


from xgboost import plot_importance
ax = plot_importance(mod)
fig = ax.figure
fig.set_size_inches(10, 30)
pyplot.show()


# In[125]:


# def get_xgb_imp(xgb, feat_names):
#     from numpy import array
#     imp_vals = xgb.booster().get_fscore()
#     print(imp_vals)
#     imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
#     total = array(imp_dict.values()).sum()
#     return {k:v/total for k,v in imp_dict.items()}

# f_score = list(mod.booster().get_fscore().values())
# print (len(f_score))
# var_imp = f_score/np.sum(f_score)
# k = df_sample_pd.shape[1]-1
# var_imp_df = pd.Series(data = var_imp, index = df_sample_pd.iloc[:, 3:k].columns))


# ### Select the most important features from the whole columns
# -- A small dataset with less columns will be returned. 
