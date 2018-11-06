
# coding: utf-8

# ### Libraries

# In[1]:


##===========================================================
### Libraries
##===========================================================

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
#import matplotlib.pyplot as plt # plots


# In[ ]:


# Load PySpark
conf = pyspark.SparkConf().setAll([('spark.executor.memory', '16g'), 
                                   ('spark.executor.instances','10')])
#*80GB of memory to work with on the Spark cluster
sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


# ### Read Dataset

# In[2]:


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


# In[3]:


df_AIRB = read_df_inferSchema("AIRB_TIMEKEY_Date")
df_AIRB.cache() # Cache data for faster reuse


# In[4]:


df_AIRB.printSchema()


# ### Subsetting the dataframe with numeric features
# 
# * Asumming that columns with string and timestamp data type are not numeric columns
# * This is not accurate but it is a first approximation
# 

# In[5]:


## subset only variables with numeric values
# Not include: timestamp, string
columnList_object = [item[0] for item in df_AIRB.dtypes if item[1].startswith('string') | item[1].startswith('timestamp')] 
columnList_object


# In[6]:


columnList_numeric = np.setdiff1d(df_AIRB.columns,columnList_object)
columnList_numeric


# In[7]:


df_numeric = df_AIRB.select(*[col(c).cast("double") for c in columnList_numeric])


# In[8]:


pd.DataFrame(df_numeric.take(5), columns=df_numeric.columns).transpose()


# In[10]:


df_numeric.count()


# ### Univariable correlation
# 
# * First analysis in raw data.
# * For imputation: NaNs values have been filled with 0

# In[14]:


## correlation using rrd.map

from pyspark.mllib.stat import Statistics
import pandas as pd

df = df_numeric
df= df.fillna(0)
col_names = df.columns
features = df.rdd.map(lambda row: row[0:])
corr_mat=Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names


# In[15]:


corr_df


# #### Another approach to verify results

# In[ ]:


## correlation 2 using VectorAssembler

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

df2 = df_numeric
df2= df2.fillna(0)
# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=df2.columns, outputCol=vector_col)
df_vector = assembler.transform(df2).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
matrix_df= matrix.collect()[0]["pearson({})".format(vector_col)].values
matrix_df


# In[ ]:


# transforming to pandas dataframe for visualization
size = len(df2.columns) 
col_names = df2.columns
corr_df2 = pd.DataFrame(matrix_df.reshape(size,size))
corr_df2.index, corr_df2.columns = col_names, col_names
corr_df2

