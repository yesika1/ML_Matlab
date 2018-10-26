
# coding: utf-8

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
import matplotlib.pyplot as plt # plots

# Load PySpark
conf = pyspark.SparkConf().setAll([('spark.executor.memory', '16g'), 
                                   ('spark.executor.instances','10')])
#*80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext


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


df = read_df_inferSchema("stg_macro_econ_HDFS")
df.cache() # Cache data for faster reuse


# In[4]:


df.printSchema()


# ##### Macroeconomic Variables Transformation

# In[168]:


##lag Example
from pyspark.sql.window import Window

w = Window().partitionBy().orderBy(col("Date"))
df.select("CA_CARealGDP", lag("CA_CARealGDP",2,0).over(w).alias("CA_CARealGDP_l1")).show(5)
#df.select(df.CA_CARealGDP, df.CA_CARealGDP_l1,).show(5)


# In[170]:


df2 = df.withColumn('CA_CARealGDP_l1', lag("CA_CARealGDP",1,0).over(w))
df2.printSchema()


# In[167]:


df2 = df.withColumn('CA_CARealGDP_l1', lag("CA_CARealGDP",1,0).over(w))
df2 = df2.select("CA_CARealGDP",df2.CA_CARealGDP - df2.CA_CARealGDP_l1).show(5)
df2


# In[195]:


# As a fucntion
import pyspark.sql.functions as F

w = Window().partitionBy().orderBy(col("Date"))
fn = F.udf(lambda col: col - col.lag(col,1,0).over(w))

df3 = df.withColumn('CA_CARealGDP_AA', fn(df.CA_CARealGDP))


# In[26]:


df2.select("CA_CARealGDP", (df2.CA_CARealGDP - lag("CA_CARealGDP").over(w)).alias('CA_CARealGDP' + '_QD')).show(5)


# In[165]:


df2.select("CA_CARealGDP", (df2.CA_CARealGDP - lag("CA_CARealGDP").over(w)).alias('CA_CARealGDP' + '_QD')).show(5)


# In[186]:


# Doing for a var and then creating the function

from pyspark.sql.window import Window
import pyspark.sql.functions as F

def QD_transformation(df2,OrderCol):
    w = Window().partitionBy().orderBy(col(OrderCol))
    df = df2.select(* ( ( col(c) - lag(col(c),1,0).over(w)).alias( c + '_QD') for c in df2.columns))
    return df

def QQ_transformation(df2,OrderCol):
    w = Window().partitionBy().orderBy(col(OrderCol))
    df = df2.select(* ( ((( col(c) - lag(col(c),1,0).over(w))-1)).alias( c + '_QQ') for c in df2.columns))
    return df

def YY_transformation(df2,OrderCol):
    w = Window().partitionBy().orderBy(col(OrderCol))
    df = df2.select(  * ( ((( col(c) / lag(col(c),4).over(w))-1)*100).alias( c + '_AA') for c in df2.columns))
    return df


# In[95]:


# Sample
## Working in a sample 
subset1= df.select(df.columns[:10])
subset1= sqlContext.createDataFrame(subset1.head(100), subset1.schema)


# In[190]:



subset2 =YY_transformation(subset1,'Date')


# In[177]:


pd.DataFrame(subset2.take(20), columns=subset2.columns).transpose()


# In[ ]:


### PANDAS VERSION


# In[318]:


# Transform dataset to df 
df_pd = df.toPandas()
df_pd = df_pd.sort_values("Date")
df_pd.head(2)


# In[322]:


# Function lag
df_pd['prev_CA_CARealGDP_lag'] = df_pd['CA_CARealGDP'].shift()
df_pd['prev_CA_CARealGDP_lag'].head()


# In[310]:


df_pd['prev_CA_CARealGDP_QD'] = df_pd['CA_CARealGDP'] - df_pd['CA_CARealGDP'].shift()


# In[311]:


df_pd['prev_CA_CARealGDP_QD'].head()


# In[312]:


df_pd.head()


# In[313]:


def transf_QF(data):
    
    for i in data.columns:
    
        if data[i].dtype != 'object':
            
            name = str(i) + '_AA'
            data[name] = data[i] - data[i].shift()
        
    return data


# In[314]:


d = transf_QF(df_pd)


# In[316]:


d


# In[ ]:


#removing rows with values greater than date
date = 5
df = df[df.VAr1 <date]


import numpy as np
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
            data[name_qq] = (data[c]-data[c].shift()) -1
            
            # Year/year difference: (current year-last year)
            name_yd = str(c)+'_yd'
            data[name_yd] = data[c]-(data[c].shift(periods=4))
           
            # Year/year growth: (current year-last year)-1
            name_yy = str(c)+'_yy'
            data[name_yy] = (data[c]-data[c].shift(periods=4)) -1 
            
            # log of QQ difference
            name_logqd = str(c)+'_logqd'
            data[name_logqd] = np.log( data[c] / data[c].shift() )
            
            # log of Quarter/Quarter growth: (current quarter-last quarter)-1
            name_logqq = str(c)+'_logqq'
            data[name_logqq] = np.log( (data[c] /data[c].shift())-1 )
            
            # log of Year/year difference: (current year-last year)-1
            name_logyd = str(c)+'_logyd'
            data[name_logyd] = np.log( data[c] /(data[c].shift(periods=4)) )
           
            # log of Year/year growth: (current year-last year)-1
            name_logyy = str(c)+'_logyy'
            data[name_logyy] = np.log( (data[c] /data[c].shift(periods=4)) -1  )
            
            
            #data[name_logQd]=np.log(data[c]/ data[c].shift()
            # negative values inside the log, which gives nan with real numbers
            #percentage *100

    return data


x =transformations(df2,'Vara')


# In[203]:




