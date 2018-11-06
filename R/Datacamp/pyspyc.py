
# coding: utf-8

# ### Libraries

# In[2]:


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


conf = sc.getConf()
execs = conf.get('spark.executor.instances')
print execs 


# In[163]:


import os
os.getcwd()


# In[168]:


get_ipython().system(u'ls')


# ### Read Dataset

# In[3]:


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


# In[4]:


df = read_df_inferSchema("AIRB_TIMEKEY_Date")
df.cache() # Cache data for faster reuse


# In[5]:


# number of observations in the dataframe
rows = df.count()
print(rows)
# 35,183,410 obs.
#  df.unpersist() # To remove the cached in memory 


# In[7]:


# number of columns in the dataframe
len(df.columns)


# In[6]:


# Display dataframe Schema
# display(df) # line by line
df.printSchema() #better visulization


# In[11]:


#Summary statistics
describe_raw = df.describe().toPandas().transpose().reset_index() # Converted & transposed the df for better visulization
describe_raw 


# In[10]:


import os
# os.chdir(r'C:\Users\ycont01\Dataiku')
#save file as csv: variable.to_csv('NewfileName.csv')

describe_raw.to_csv(r'C:\Users\ycont01\Desktop\describeAIRB.csv',index=True)


# In[9]:


# View the first five observations
# df.take(5) # line by line 
pd.DataFrame(df.take(5), columns=df.columns).transpose() # better visulization with pandas


# ### Summary Statistics in Raw Data

# In[4]:


##===========================================================
### Summary Statistics in Raw Data
##===========================================================


# Explore if the target is balanced
df.groupBy('y').count().show()
# Unbalanced: 0= 34990640, 1=192770 
# df.cube("y").count().show()


# In[11]:


#Summary statistics for numeric variables
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1] == 'double']
df.select(numeric_features).describe().toPandas().transpose()


# In[12]:


#Summary statistics
describe_raw = df.describe().toPandas().transpose().reset_index() # Converted & transposed the df for better visulization
describe_raw


# ### Data Preparation

# #### Missing Values

# In[18]:


##===========================================================
### Data preparation
##===========================================================


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Missing Values
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Some examples of data distribution
df.groupBy('APPL_DUP_PAD_TYPE').count().show()


# In[17]:


df.groupBy('MORT_BAL_AMT_TOT_AMT').count().show()
# just one disctinct value


# In[56]:


df.agg({'TIME_DIM_KEY': 'min'}).show()


# In[ ]:


df.agg({'APPL_LAST_UPDATE_DT': 'min'}).show()


# In[11]:


df.groupBy('BMO_CB_MORT_WORST_RT').count().show()


# ##### Total missing values

# In[119]:


from pyspark.sql.functions import col, sum

# Calculate the total number of missing value for each column variable.
df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).toPandas().transpose() 


# ##### Percentage of missing

# In[159]:


mis_val =df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns))                     
                    
for field in missing.schema.fields:
    if str(field.dataType) in ['DoubleType', 'FloatType', 'LongType', 'IntegerType', 'DecimalType']:
        name = str(field.name)
        mis_val = mis_val.withColumn(name, col(name)/rows)
        
mis_val.toPandas().transpose()


# In[23]:


# Add another mis_val_percent column to show the percentages of missing
from pyspark.sql.functions import col, sum
mis_val = df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).toPandas().transpose()
mis_val.insert(1, 'Total missing value', mis_val[0] )
mis_val.insert(2, 'Percentage of missing value', 100 * mis_val[0] / rows )

# Sort the table by ascending percentages of missing
mis_val.sort_values(['Percentage of missing value'], ascending=[False], inplace=True)

print (mis_val)


# In[ ]:


mis_val.plot(kind='barh',y='Total missing value',colormap='winter_r')


# In[31]:


mis_val.plot(kind='bar',y='Total missing value')
# Big changed with data less than 10M


# ###### Idenfitying columns with only missing values

# In[106]:


sdf =df
for col in sdf.columns:
    if (sdf.filter(func.isnan(func.col(col)) == True).count() == sdf.select(func.col(col)).count()):
        sdf = sdf.drop(col) 
    if (sdf.filter(func.col(col).isNull()).count() == sdf.select(func.col(col)).count()):
        sdf = sdf.drop(col)
        
# IT takes to long, but it is working 


# In[109]:


# variables to drop because all the values are null: 20 variables
import numpy as np
columns_not_list = np.setdiff1d(df.columns,sdf.columns)
print(columns_not_list)


# In[21]:


# Verifying before dropping rows
columns_not_list = ['APPL_DUP_PAD_TYPE', 'APPL_REC_TYP_CD', 'DISBLTY_FG', 'INS_COVERAGE_IND',
 'INVALID_ACT_NO', 'MEMB_NO', 'NAS_CO3_SEX_CD', 'NAS_OPEN_ACT_NO',
 'NAS_PRI_SEX_CD', 'NAS_SEC_SEX_CD', 'NBA_CO3_AGE_MNTH_CNT', 'NBA_CO3_DOB_DT',
 'NBA_CO3_TITLE_NM', 'RGON_CD' ,'STRATA_GDSR_TAG_TXT', 'STRATA_TDSR_TAG_TXT',
 'USER_CAP_LVL' ,'USER_NM' ,'acnm', 'tran']
describe_raw = df.describe(columns_not_list).toPandas().transpose().reset_index()
describe_raw


# In[110]:


len(columns_not_list)


# In[6]:


## creating the function:
def missing_values(dataframe):
    '''Display percentage of missing values within each feature
    Arg: spark dataframe
    return: pandas dataframe
    '''
    rows = dataframe.count()
    mis_val = dataframe.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in dataframe.columns)).toPandas().transpose()
    mis_val.insert(1, 'Percentage of missing value', 100 * mis_val[0] / rows)
    mis_val.sort_values(['Percentage of missing value'], ascending=[False], inplace=True)
    mis_val = mis_val.rename(columns={0:'count'})

    return mis_val


# In[13]:


from time import time
t0 =time()
df_missing = missing_values(df)
tt = time()-t0
print('Query performed in {} seconds'.format(tt))


# In[15]:


def missing_thresholds(dataframe, thres):
    """return a list of columns with the threshold greater than the percentage of missing values
        arg: int, dec
        return: int, dataframe
    """
    thres_df = dataframe[dataframe['Percentage of missing value'] < thres]
    length = len(thres_df )
    return length, thres_df


# In[ ]:


len(missing_thresholds(df_missing, 100)) # 20 var. dropped


# In[16]:


t0 =time()
leng, col_noMissing = missing_thresholds(df_missing, 100)
tt = time()-t0
print('Query performed in {} seconds'.format(tt))


# In[39]:


col_names = col_noMissing.transpose().columns


# ###### Idenfitying columns as identifiers (only unique values)

# In[17]:


# Identifying columns with all unique values

def dropIdentifier(df):
    # Iterate over each column in the DF
    rows= df.count()
    dropped =[]
    for col in df.columns:
        # Get the distinct values of the column
        unique_val = df.select(col).distinct().count()
        # See whether the sum of the unique value is the column size
        if unique_val == rows:
            print("Dropping " + col + " because ALL values are unique.")
            dropped.append(col)
            #df = df.drop(col)
        
    #return(df)
    return(dropped)


# In[18]:


list_drop_unique= dropIdentifier(df)
list_drop_unique # no complete unique values bc dataframe is after joining mago

## pending to check before the final join dataset


# #### Create New dataframe without 100% missing values

# In[46]:


# Create a new Spark Dataframe which only contains columns not having missing value percentages equal to 100. 
rows = 35183410 #df.count()
mis_val = df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df.columns)).toPandas().transpose()
mis_val.insert(1, 'Total missing value', mis_val[0] )
mis_val.insert(2, 'Percentage of missing value', 100 * mis_val[0] / rows )
# df_new = df.select([c for c in df.columns if mis_val.loc[c, 'Percentage of missing value']<20])
df_new = df.select([c for c in df.columns if mis_val.loc[c, 'Percentage of missing value']<100.0])
# Check the number of remaining columns in the new Spark Dataframe
len(df_new.columns)


# #### Function for data types

# In[47]:


## Function for Data types
# Import all from `sql.types`
from pyspark.sql.types import *

# Write a custom function to convert the data type of DataFrame columns
def convertColumn(df, names, newType):
    for name in names: 
        try: 
            df = df.withColumn(name, df[name].cast(newType))
        except:
            pass
    return df 


# In[48]:


# Running: Create a list to Assign all column names to `columns`
columns = df_new.columns

# Call function & Conver the `df` columns to `FloatType()`
# for now transforming all columns
df = convertColumn(df, columns, FloatType())


# In[49]:


df.printSchema()


# ## Feature Selection

# In[ ]:


##==================================
## Feature Selection
##==================================


## Random Forest



# ## subsetting data

# In[ ]:


## for testing before applying to the entire dataframe


# In[ ]:


## Working in a sample 
subset1= df.select(df.columns[:10])
subset1= sqlContext.createDataFrame(subset1.head(100), subset1.schema)


# In[164]:


#subset1.describe().toPandas().transpose().to_csv('mytest1.csv')


# In[ ]:


subset1.printSchema()


# In[177]:


stats =subset1.describe().toPandas().transpose().reset_index()


# In[178]:


stats[:2]

