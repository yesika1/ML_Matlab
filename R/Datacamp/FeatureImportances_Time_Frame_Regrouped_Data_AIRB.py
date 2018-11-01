
# coding: utf-8

# In[3]:


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
conf = pyspark.SparkConf().setAll([('spark.executor.memory', '6g'), 
                                   ('spark.executor.instances','10')])
#80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


# In[2]:


# Example: Read the descriptor of a Dataiku dataset
mydataset = dataiku.Dataset("AIRB_TIMEKEY")
# And read it as a Spark dataframe
df = dkuspark.get_dataframe(sqlContext, mydataset)


# In[9]:


# the number of columns in the dataframe
len(df.columns)


# In[9]:


# get a list of the column names for this dataframe
df.columns


# In[10]:


# Examples: to count the total records in the dataframe

rows = df.count()
print(rows)


# In[11]:


# Show the content of the "TIME_DIM_KEY" column
# df.select('TIME_DIM_KEY').show()

# Show the distinct "TIME_DIM_KEY" in the dataframe
df.select('TIME_DIM_KEY').distinct().show()


# In[12]:


# Count how many distinct "TIME_DIM_KEY" in the dataframe
df.select('TIME_DIM_KEY').distinct().count()


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

# In[18]:


from pyspark.sql.functions import col

# 7 snapshots (2009 Oct 31 to 2011 April 30)
snapshot_list = ['37547', '37550', '37553', '37556', '37559', '37562', '37565']

# Choose/filter those snapshots and return a filtered dataframe
df_snapshots = df.where(col('TIME_DIM_KEY').isin(snapshot_list))

# Total records in the filtered dataframe
df_snapshots.count()


# In[11]:


from pyspark.sql.functions import col

# 7 snapshots (31/10/2006 to 30/04/2008)
snapshot_list = ['37511', '37514', '37517', '37520', '37523', '37526', '37529']

# Choose/filter those snapshots and return a filtered dataframe
df_snapshots = df.where(col('TIME_DIM_KEY').isin(snapshot_list))

# Total records in the filtered dataframe
df_snapshots.count()


# In[23]:


# take the whole data frames
df_snapshots = df


# The 7 snapshots data is about 5.3 million rows; 
# the data size is still too large for Pandas dataframe

# In[19]:


get_ipython().run_cell_magic(u'time', u'', u'# df_snapshots.toPandas()  \n# failed. \n# Error message: Total size of serialized results of 246 tasks (2.0 GB) is bigger than spark.driver.maxResultSize (2.0 GB)')


# In[20]:


# Group by the target variable "y", count the records in the groups
df_snapshots.groupBy('y').count().show()


# The ratio between the Non-defaults : Defaults is about 200: 1, for the 7 snapshots of the data. 

# In[4]:


# drop duplicate rows from the df_snapshots
df_snapshots.dropDuplicates().count()


# There are no duplicate rows, as the total number of remaining rows keeps the same: 5290482 rows. 

# In[17]:


# Example: View the first 3 rows of the dataframe
df_snapshots.take(3)


# In[18]:


# Print the Schema of the dataframe
df_snapshots.printSchema()


# In[24]:


df_snapshots = df_snapshots.select(*[col(c).cast("double") for c in df_snapshots.columns])
# df_sample = df_new_sample.select(*[col(c) for c in df_new_sample.columns])

# Drop Channel, Status and Role as they are character type
df_snapshots = df_snapshots.drop('Channel','Status', 'Role')


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

# In[1]:


missing_table = missing_values(df_snapshots)
# return a table summarizing the missing percentages for each variables

missing_table


# ## Take a random sample out of the **df_snapshots** dataframe 

# In[25]:


# Draw a 1% randomly sample rows from the base Dataframe
df_sample = df_snapshots.sample(False, fraction = 0.01, seed=42)


# In[26]:


df_sample.count()


# This is about 264781/5290482 = 5% of the 7 snapshots of data.

# In[27]:


# change the type of the element in the dataframe from string into double. 
df_sample = df_sample.select(*[col(c).cast("double") for c in df_sample.columns])

# Drop Channel, Status and Role as they are character type
df_sample = df_sample.drop('Channel','Status', 'Role')


# ### Convert this sample Pyspark dataframe into a Pandas dataframe

# In[28]:


# convert to Pandas dataframe
df_sample_pd =df_sample.toPandas()

# show the shape of this Pandas dataframe
df_sample_pd.shape


# In[29]:


df_sample_pd.head(20)


# In[30]:


# check the type of the dataframe

print (df_sample_pd.dtypes)


# ### Label encoder for the categorical variables

# In[31]:


# get the frequency counts of the data type of the elements in a variable over the entire Pandas dataframe.
dtypeCount = [df_sample_pd.iloc[:, i].apply(type).value_counts() for i in range (df_sample_pd.shape[1])]


# In[32]:


dtypeCount


# In[33]:


import numpy as np
from sklearn import preprocessing

categorical_feats = [f for f in df_sample_pd.columns if df_sample_pd[f].dtype == 'object']
print (len(categorical_feats))  

lb = preprocessing.LabelEncoder()
for colu in categorical_feats:
    lb.fit(list(df_sample_pd[colu].values.astype('str')))
    df_sample_pd[colu] = lb.transform(list(df_sample_pd[colu].values.astype('str')))

  


# In[34]:


df_sample_pd


# ### use XGBoost to rank the feature importances for all variables

# In[35]:


from xgboost import XGBClassifier
# from xgboost import gbtree
model = XGBClassifier()
X = df_sample_pd.iloc[:, 2:433]
y = df_sample_pd.iloc[:, 1]
mod = model.fit(X, y) 


# In[36]:


from xgboost import plot_importance
ax = plot_importance(mod)
fig = ax.figure
fig.set_size_inches(10, 30)
pyplot.show()


# In[21]:


# def get_xgb_imp(xgb, feat_names):
#     from numpy import array
#     imp_vals = xgb.booster().get_fscore()
#     print(imp_vals)
#     imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
#     total = array(imp_dict.values()).sum()
#     return {k:v/total for k,v in imp_dict.items()}

f_score = list(mod.booster().get_fscore().values())
print (len(f_score))
var_imp = f_score/np.sum(f_score)
k = df_sample_pd.shape[1]-1
# var_imp_df = pd.Series(data = var_imp, index = df_sample_pd.iloc[:, 3:k].columns))


# In[22]:


len(var_imp)


# In[80]:


get_xgb_imp(mod, list(df_sample_pd.iloc[:, 3:436].columns))


# In[134]:


df_sample_pd.head()
df_sample_pd.shape[1]


# In[144]:



k = df_sample_pd.shape[1]-1
df_sample_pd.iloc[:, 3:k].columns()

