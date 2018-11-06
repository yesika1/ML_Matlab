
# coding: utf-8

# In[1]:


get_ipython().magic(u'pylab inline')


# ### CCAP_DEV_HDFS as pyspark dataframe

# In[4]:


get_ipython().magic(u'pylab inline')
import dataiku
import dataiku.spark as dkuspark
import pyspark
from pyspark.sql import SQLContext

from pyspark.sql.types import StringType # to return strings in functions
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer # to transform categorical columns to label
import pyspark.sql.functions as func
from pyspark.sql.functions import isnan

import pandas as pd #Pandas data frame is prettier than Spark DataFrame.show().
import numpy as np #vector operations
import matplotlib.pyplot as plt # plots

# Load PySpark
conf = pyspark.SparkConf().setAll([('spark.executor.memory', '16g'), 
                                   ('spark.executor.instances','10')])
#*80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


# In[5]:


##===========================================================
### Read Dataset
##===========================================================


# Read the descriptor of the Dataiku dataset
# dataset = dataiku.Dataset("AIRB")
# And read it as a Spark dataframe
# df = dkuspark.get_dataframe(sqlContext, dataset)


#Load Spark data frame with inferred schema
def read_df_inferSchema(dataset):
    ''' Load Spark dataframe with inferred schema
    arg: dataset, it is the name of the file'''
    
    dataset = dataiku.Dataset(dataset)

    location = dataset.get_location_info()
    files = dataset.get_files_info()
    schema = dataset.read_schema()

    directory = location['info']['path']
    global_paths = files['globalPaths']
    hdfs_paths = [g['path'] for g in global_paths]
    full_paths = [ directory + p for p in hdfs_paths]
    col_names = [s['name'] for s in schema]

    df = sqlContext.read.load(full_paths, format = 'com.databricks.spark.csv', inferSchema='true',delimiter='\t').toDF(*col_names)
    return df


# In[6]:


df = read_df_inferSchema("CCAP_DEV_HDFS")
df.cache() # Cache data for faster reuse


# In[7]:


df.printSchema()


# In[8]:


df.select('book').groupBy('book').count().show()


# In[9]:


df.select('accept','book').groupBy('accept','book').count().show()


# In[10]:


df.select('adjud_cd','accept','book').groupBy('adjud_cd','accept','book').count().show()


# In[12]:


df.count()


# In[15]:


df2 = df.na.drop(subset=['book'])


# In[16]:


df2.count()


# In[17]:


df2.select('adjud_cd','accept','book').groupBy('adjud_cd','accept','book').count().show()


# In[18]:


df3 = df2


# In[22]:


df3 = df3.withColumn('adjud_cd',               when(df['accept'] == 1, 'A').otherwise('R'))


# In[23]:


df3.select('adjud_cd','accept','book').groupBy('adjud_cd','accept','book').count().show()


# In[24]:


df3.count()


# ### CCAP_DEV_HDFS as pandas dataframe

# In[2]:


import dataiku
from dataiku import pandasutils as pdu
import pandas as pd


# In[3]:


# Read the dataset as a Pandas dataframe in memory
# Note: here, we only read the first 100K rows. Other sampling options are available
dataset_CCAP_DEV_HDFS = dataiku.Dataset("CCAP_DEV_HDFS")
df = dataset_CCAP_DEV_HDFS.get_dataframe()


# ### Performance_HDFS_TARGET

# In[ ]:


dataset_PERFORMANCE_HDFS_TARGET = dataiku.Dataset("PERFORMANCE_HDFS_TARGET")
df_performance = dataset_PERFORMANCE_HDFS_TARGET.get_dataframe()


# In[ ]:


actsysid_in_CCAP_dev = df.isin(df_performance['actsysid'])


# In[ ]:


for i in actsysid_in_CCAP_dev.columns:
    
    if any(actsysid_in_CCAP_dev[actsysid_in_CCAP_dev[i] == True]):
        print(i, len(actsysid_in_CCAP_dev[actsysid_in_CCAP_dev[i] == True]))
        index = actsysid_in_CCAP_dev[actsysid_in_CCAP_dev[i] == True].index


# In[ ]:


index


# In[ ]:


df.iloc[[303499, 303500, 336277]]['cls_doc_file_nbr']


# In[8]:


import featuretools as ft

