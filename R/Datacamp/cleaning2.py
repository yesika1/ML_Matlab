
# coding: utf-8

# In[1]:



# Load PySpark
from datetime import datetime
import dataiku
import dataiku.spark as dkuspark
from dataiku import pandasutils as pdu
import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType # to return strings in functions
from pyspark.sql.functions import udf #for user defined functions
from pyspark.ml.feature import StringIndexer # to transform categorical columns to label
from pyspark.sql.functions import unix_timestamp, to_date
from pyspark.sql.functions import isnan, when, count, col, trim, stddev
import pandas as pd #Pandas data frame is prettier than Spark DataFrame.show().
import numpy as np #vector operations
import matplotlib.pyplot as plt # plots
from pyspark.sql.types import DateType
from pyspark.sql import functions as F #Spark function


# In[2]:


# Load PySpark
conf = pyspark.SparkConf()
#*80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


# In[3]:


# Read recipe inputs
ccap_DEV_HDFS = dataiku.Dataset("CCAP_DEV_HDFS")
df = dkuspark.get_dataframe(sqlContext, ccap_DEV_HDFS)


# In[4]:


def drop_missing(dataframe):
    rows = dataframe.count()
    temp = dataframe.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dataframe.columns]).toPandas()
    columns = [c for c in temp.columns if temp[c][0] == rows ]

    return  dataframe.drop(*list(columns))


# In[5]:


df1 = drop_missing(df)


# In[6]:


def drop_dup(dataframe):
    return dataframe.drop_duplicates()


# In[7]:


df1 = drop_dup(df1)


# In[8]:


def drop_distinct(dataframe):

    temp = dataframe.select([stddev(c).alias(c) for c in dataframe.columns]).toPandas()
    columns = [c for c in temp.columns if temp[c][0] == 0.0 ]

    return dataframe.drop(*list(columns))


# In[9]:


df2 = drop_distinct(df1)


# In[10]:


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a SparkSQL dataframe
ccap_DEV_MODELING_df = df2 # For this sample code, simply copy input to output

# Write recipe outputs
ccap_DEV_MODELING = dataiku.Dataset("CCAP_DEV_MODELING")
dkuspark.write_with_schema(ccap_DEV_MODELING, ccap_DEV_MODELING_df)

