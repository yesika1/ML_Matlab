
# coding: utf-8

# ## Variable Transformation and Data Imputation for AIRB Candidate Variables
# #### 1. Select important features from AIRB_TIMEKEY_Date variables (only PD eligible population)
# #### 2. Load and map the macroeconomic factors into the main dataset
# #### 3. Draw a sample data and Convert to Pandas dataframe
# #### 4. Variable Transformation 
# #### 5. Data Imputation
# 

# In[2]:


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


# ####  Load AIRB_TIMEKEY_Date dataset as a PySpark Dataframe

# In[3]:


# Example: Read the descriptor of a Dataiku dataset
mydataset = dataiku.Dataset("AIRB_TIMEKEY_Date")
# And read it as a Spark dataframe
df = dkuspark.get_dataframe(sqlContext, mydataset)


# In[5]:


# Example: Get the count of records in the dataframe
df.count()


# In[4]:


# the number of columns in the dataframe
len(df.columns)


# #### Exclude PD Ineligible Population

# In[5]:


## exclude ineligible PD population by "STATUS"
from pyspark.sql.functions import col

# CR (Credit Risk) and ED (Early Delinquency)
eligible_list = ['CR', 'ED']

# Choose/filter the PD eligible population and return a filtered dataframe
df = df.where(col('STATUS').isin(eligible_list))

# Total records in the filtered dataframe
# df.count() 


# #### Load Macroeconomic Factors dataset as a PySpark dataframe

# In[6]:


# Example: Read the descriptor of a Dataiku dataset
mydataset2 = dataiku.Dataset("stg_macro_econ_transform2")
# And read it as a Spark dataframe
df_macro = dkuspark.get_dataframe(sqlContext, mydataset2)


# In[7]:


# select a few macroeconomic factors from the whole dataset
df_macro_few = df_macro.select("Date", "CA_UNEMPLOYMENT_RATE", "CA_CARealGDP", "CA_REALGDP_YY", "CA_REALGDP_QA", 
                               "CA_REALGDP_RE", "CA_HOUSEPRICE_YY", "CA_HOUSEPRICE_QA", "CA_AVG_BENCHMARK_CAP_RATE")

len(df_macro_few.columns)


# ### 1. Select candidate variables from the main dataset
# #### a. Choose Time Frame (most recent 5 years data)
# #### b. Select candidate variables/columns
# #### c. Draw a random sample data and convert to Pandas dataframe

# In[8]:


from pyspark.sql.functions import col

# 20 snapshots (31/07/2012 to 30/04/2017)
snapshot_list = ['37580', '37583', '37586', '37589', '37592', '37595', '37598', '37601', '37604', '37607', '37610', 
                 '37613', '37616', '37619', '37622', '37625', '37628', '37631', '37634', '37637']

# 7 most recent snapshots 
# snapshot_list = ['37619', '37622', '37625', '37628', '37631', '37634', '37637']

# Choose/filter those snapshots and return a filtered dataframe
df_snapshots = df.where(col('TIME_DIM_KEY').isin(snapshot_list))

# Whole data 10 years data
#df_snapshots = df

# Total records in the filtered dataframe
df_snapshots.count()


# In[9]:


# Draw a 5% randomly sample rows from the base Dataframe
df_sample = df_snapshots.sample(False, fraction = 0.05, seed=42)


# In[10]:


df_sample.count()


# In[19]:


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


# In[20]:


missing_table = missing_values(df_snapshots)
# return a table summarizing the missing percentages for each variables

missing_table


# In[11]:


# change the type of the element in the dataframe from string into double. 
df_sample = df_sample.select(*[col(c).cast("double") for c in df_sample.columns])

# Drop Channel, Status and Role as they are character type
df_sample = df_sample.drop('COUNT', 'Channel','Status', 'Role', 'TIME_DIM_KEY', 'Date', 'SP_ACT_KEY', 
                           'pdclass', 'PD_SCR_VAL', 'APPL_DIM_KEY')

#df_sample.printSchema()


# ### data exploration for the sample data
# -- check the filling content of each column

# In[22]:


df_sample.select('BMO_TDSR_7_PCT').distinct().count()


# ### df_sample data Convert to Pandas dataframe: be prepared for xgboost ranking

# In[12]:


# convert to Pandas dataframe
df_sample_pd =df_sample.toPandas()

# show the shape of this Pandas dataframe
df_sample_pd.shape


# In[13]:


df_sample_pd.head(5)


# In[14]:


# get the frequency counts of the data type of the elements in a variable over the entire Pandas dataframe.
dtypeCount = [df_sample_pd.iloc[:, i].apply(type).value_counts() for i in range (df_sample_pd.shape[1])]

dtypeCount


# #### Label Encoder the categorical variables

# In[15]:


import numpy as np
from sklearn import preprocessing

categorical_feats = [f for f in df_sample_pd.columns if df_sample_pd[f].dtype == 'object']
print (len(categorical_feats))  

lb = preprocessing.LabelEncoder()
for colu in categorical_feats:
    lb.fit(list(df_sample_pd[colu].values.astype('str')))
    df_sample_pd[colu] = lb.transform(list(df_sample_pd[colu].values.astype('str')))


# In[16]:


df_sample_pd.head(10)


# In[17]:


length = df_sample_pd.shape[1]
print (length)


# ### Use XGBOOST to rank feature importance

# In[18]:


from xgboost import XGBClassifier

model = XGBClassifier()
X = df_sample_pd.iloc[:, 2:(length-1)]
y = df_sample_pd.iloc[:, 1]
mod = model.fit(X, y) 


# In[19]:


from xgboost import plot_importance
ax = plot_importance(mod)
fig = ax.figure
fig.set_size_inches(10, 30)
pyplot.show()


# In[20]:


# get a dictionary of the important features and their f-score. 
#importances = mod.booster().get_fscore()

importances = mod._Booster.get_fscore()

#import operator
# sort the order of the dictionary by the value; descending order
# sorted_importances = sorted(importances.items(), key=operator.itemgetter(1), reverse=True)

# sorted_importances


# In[21]:


# Create an empty tuple to contain the important features
important_feat = ()

for name, value in importances.items():
    #print ('the name of the importances dictionary', name)
    important_feat += (name,)

# Return a tuple containing important features
important_feat


# ### df_snapshots : select important features based on xgboost ranking results

# In[22]:


# Select important features
df_new = df_snapshots.select (*important_feat)

# how many important features have been selected. 
len(df_new.columns)


# In[23]:


# Missing value percentages for the smaller dataset

missing_table2 = missing_values(df_new)
# return a table summarizing the missing percentages for each variables

missing_table2


# In[25]:


# Add the target variable 'y' into the list
important_feat_y = ('y',) + important_feat

df_whole = df.select(*important_feat_y)

# missing_table2 = missing_values(df_whole)
# return a table summarizing the missing percentages for each variables

# missing_table2


# In[26]:


# change the type of the element in the dataframe from string into double. 
df_new = df_new.select(*[col(c).cast("double") for c in df_new.columns])

# change the type of the element in the dataframe from string into double. 
df_whole = df_whole.select(*[col(c).cast("double") for c in df_whole.columns])


# In[27]:


df_whole.printSchema()


# In[28]:


df_whole.select('CRBURS').show()


# ### df_new : the new PySpark dataframe (sample data) that only contains important features
# ### df_whole : the new PySpark dataframe (whole PD eligible population) that only contains important features plus target "y"
# 
# 1. Variable Transformation/Binning in PySpark using Bucketizer
# 2. Missing Value Imputation in PySpark using Imputer

# In[79]:


df_new.select('CRBURS').describe().show()


# In[30]:


# from pyspark_dist_explore import hist

#fig, ax = plt.subplots()
#hist(ax, df_whole, bin = 2, color =['red'])

bins, counts = df_whole.select('nc_oldest_mob').rdd.flatMap(lambda x:x).histogram(20)
plt.hist(bins[:-1], bins=bins, weights=counts)


# In[35]:


bins, counts = df_whole.select('CRBURS').rdd.flatMap(lambda x:x).histogram(5)
plt.hist(bins[:-1], bins=bins, weights=counts)


# In[37]:


from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(splits=[654, 716, 765, float('inf') ],inputCol="CRBURS", outputCol="CRBURS_buckets")
df_buck = bucketizer.setHandleInvalid("keep").transform(df_whole)

df_buck.select('CRBURS', 'CRBURS_buckets').show()


# In[39]:


from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols=['CRBURS'], outputCols=['out_CRBURS'])
#imputer.setStrategy("median").
mean_model = imputer.fit(df_whole)
df_impute = mean_model.transform(df_whole)
df_impute.select('CRBURS', 'out_CRBURS').show(100)


# 3. Variable Transformation/Binning in Pandas using Pandas.cut()
# 4. Missing value Imputation in Pandas using
