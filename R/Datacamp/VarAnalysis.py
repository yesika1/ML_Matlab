
# coding: utf-8

# In[5]:


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


# In[4]:


distFile = sc.textFile("AIRB_TIMEKEY_Date")


# In[3]:


# Example: Read the descriptor of a Dataiku dataset
mydataset = dataiku.Dataset("AIRB_TIMEKEY_Date")
# And read it as a Spark dataframe
df = dkuspark.get_dataframe(sqlContext, mydataset)


# In[3]:


df.cache()


# In[4]:


# Example: View the first 10 rows of the dataframe
# df.take(10)


# In[5]:


# display(df)


# In[ ]:


# the number of columns in the dataframe
len(df.columns)


# In[7]:


# df.columns


# In[11]:


# # Examples: to count the total records in the dataframe
# rows = df.count()
# print(rows)


# ## Data Exploration

# In[4]:


# drop duplicated rows
df.dropDuplicates()

# drop keys
# df.drop('sp_act_key','pd_scr_val', 'appl_dim_key','time_dim_key')


# In[6]:


# Draw a 1% randomly sample rows from the base Dataframe
df_sample = df.sample(False, fraction = 0.01, seed=42)
# df_sample.show()
# rows = df_sample.count()


# In[7]:


from pyspark.sql.functions import col



snapshot_list = ['37580', '37583', '37586', '37589', '37592', '37595', '37598', '37601', '37604', '37607', '37610', 
                '37613', '37616', '37619', '37622', '37625', '37628', '37631', '37634', '37637']
    
    # Choose/filter those snapshots and return a filtered dataframe
df_snapshots = df_sample.where(col('TIME_DIM_KEY').isin(snapshot_list))

df_cols = df_snapshots.select('bmo_assets', 'BMO_TDSR_3_PCT', 'BOOK_DT', 'CP_MC_AP_REASON_CD', 'CRBURS', 'acctage', 
                 'Curr_DLQ','cyc2x9mp', 'utiliznp', 'WORST3M_pdead')
# Total records in the filtered dataframe
df_cols.count()


# In[8]:


# import seaborn as sns
# df_sample.show()
# from pyspark.sql.functions import col
# import plotly.plotly as py
# import plotly.graph_objs as go


# x = df_sample.select("bmo_assets")
# y = x.fillna({'bmo_assets':0})
# histogram = y.select('bmo_assets').rdd.flatMap(lambda x: x).histogram(11)
# print(df_sample.head(10).count())
df_cols_numeric = df_cols.toPandas().apply(pd.to_numeric, errors='coerce')

# pd.to_numeric(df_cols, errors='coerce')

# df_sample.toPandas().hist(column='bmo_assets', bins=100)
# y.select('bmo_assets').rdd.flatMap(lambda x: x).histogram(11)


# df_sample_pd = df_sample.select(df_sample['bmo_assets'].cast("double")).toPandas()
# df_sample_pd.hist(column='bmo_assets', bins=100)

# df_sample_pd = df_sample.select('bmo_assets').toPandas()
# df_sample_pd
# df_sample_pd.hist(column='bmo_assets', bins=50)

# df_sample_pd['bmo_assets'].plot.hist()
# df_sample['bmo_assets'].plot.hist()


# In[11]:


df_cols_numeric


# ## Variable Distribution starts here

# In[14]:


import seaborn as sns
df_cols_numeric.hist(column='bmo_assets', bins=10)
# df_cols_numeric_dropped_na = df_cols_numeric.dropna(subset=['bmo_assets'])
# print(len(df_cols_numeric_dropped_na))
# print(df_cols_numeric)
# print(df_cols_numeric.dtypes)
# sns.distplot(df_cols_numeric_dropped_na['bmo_assets'],bins=100, norm_hist=False)


# In[108]:


df_cols_numeric.hist(column='BMO_TDSR_3_PCT')


# In[109]:


# Ignore for now - no numerical values
# df_cols_numeric.hist(column='BOOK_DT')


# In[110]:


df_cols_numeric.hist(column='CP_MC_AP_REASON_CD')


# In[111]:


df_cols_numeric.hist(column='CRBURS')


# In[112]:


df_cols_numeric.hist(column='Curr_DLQ')


# In[113]:


df_cols_numeric.hist(column='cyc2x9mp')


# In[114]:


df_cols_numeric.hist(column='utiliznp')


# In[115]:


df_cols_numeric.hist(column='WORST3M_pdead')


# In[116]:


df_cols_numeric.hist(column='acctage')


# <img src="image name or linkf" alt="" height="" width="">

# ## Missing Value Percentages for each variable

# In[26]:


from pyspark.sql.functions import col, sum

# Calculate the total number of missing value for each column variable.
mis_val = df_sample.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in df_sample.columns)).toPandas().transpose() 
# converted the Spark dataframe into Pandas dataframe
# transpose the Pandas dataframe for better visulization


# In[27]:


# Add another mis_val_percent column to show the percentages of missing

mis_val.insert(1, 'Percentage of missing value', 100 * mis_val[0] / rows )

# Sort the table by ascending percentages of missing
mis_val.sort_values(['Percentage of missing value'], ascending=[False], inplace=True)

print (mis_val)


# In[ ]:


# df_snapshots.groupby(['STATUS', 'y'])['ACTSYSID'].agg('sum')
# Need to add outstanding balances and eligible for PD Sampling column




# In[ ]:


#mis_val.insert(0, 'Variable label', mis_val.index.values)
mis_val.loc[mis_val['Percentage of missing value']>0.0] 
# only keep non-zero missing for the mis_val dataframe


# In[ ]:


# Create a new Spark Dataframe which only contains columns having missing value percentages lower than 20% or equal to zero. 
# df_new = df.select([c for c in df.columns if mis_val.loc[c, 'Percentage of missing value']<20])
df_new = df.select([c for c in df.columns if mis_val.loc[c, 'Percentage of missing value']<=0.0])
# Check the number of remaining columns in the new Spark Dataframe
len(df_new.columns)


# In[ ]:


# drop duplicate rows; return a new dataframe with duplicate rows removed. 
df_new.dropDuplicates()

# drop rows with null value 
df_new2 = df_new.dropna()


# ## Divide the dataset into "training" and "testing" sets by the "role". 

# In[ ]:


#  df_new_train = df_new.select([c for c ])


# In[ ]:


# Take a sample dataframe from the base Dataframe
# df_new_sample = df_new2.sample(False, 0.001, 42)
df_new_sample = df_new2


# In[ ]:


from sklearn import preprocessing

categorical_feats = [f for f in df_new_sample.columns if df_new_sample[f].dtype is 'object']

lb = preprocessing.LabelEncoder()
for colu in categorical_feats:
    lb.fit(list(df_new_sample[colu].values.astype('str')))
    df_new_sample[colu] = lb.transform(list(df_new_sample[colu].values.astype('str')))

# len(categorical_feats)    


# In[ ]:


#df_new_sample.select('UTILIZN').distinct().show()
df_new_sample.select('UTILIZN').show()
#df_new_sample.select('y')
# df_new_sample.drop('ACTSYSID','y', 'Role').collect()


# In[ ]:



df_new4= df_new_sample.select(df.y.cast("double"))

#len(df_new4.columns)
#df_new4.count()


# In[ ]:


# features excluding those three columns;
df_new2 = df_new_sample.drop('ACTSYSID','y', 'Role')

# change the schema for every column; to get float type for all columns;
# df_new3 = df_new2.select(*[col(c).cast("double") for c in df_new2.columns]).toPandas() 
#df_new3.shape

#df_new3 = df_new2.select(*[col(c).cast("double") for c in df_new2.columns])
#df_new5 = df_new_sample.select(*[col(c).cast("double") for c in df_new_sample.columns])


# In[ ]:


# df_new_sample.printSchema()

df_sample = df_new_sample.select(*[col(c).cast("double") for c in df_new_sample.columns])
# df_sample = df_new_sample.select(*[col(c) for c in df_new_sample.columns])

# Drop Channel, Status and Role as they are character type
df_sample = df_sample.drop('Channel','Status', 'Role')

# df_sample.select('Role').show()
df_sample.show()
df_sample['Role']
df_sample.printSchema()
# df_sample.show()


# In[ ]:


df_sample.select('ACTSYSID').distinct().count()
# df_sample.select('ACTSYSID').show()


# In[ ]:


# df_sample.select('TIME_DIM_KEY').show()
df_sample.select('TIME_DIM_KEY').distinct().show(43)


# In[ ]:


df_sample.select('TIME_DIM_KEY').distinct().count()


# In[ ]:


# from pyspark.sql import Window
# windowSpec = Window.partitionBy(df_sample["ACTSYSID"]).orderBy(df_sample["TIM_DIM_KEY"])

# filter rows that have TIM_DIM_KEY = 37571
df_snapshot = df_sample.filter(df_sample['TIME_DIM_KEY']=='37520')


# In[ ]:


df_snapshot.count()


# In[ ]:


df_sample = df_snapshot


# ## Transform our data using the VectorAssembler function to asingle column where each row of the Dataframe contains a feature vector. 

# In[ ]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


ignore = ['ACTSYSID','y']
lista = [x for x in df_sample.columns if x not in ignore]
counta = df_sample.count()
print(counta)
vec_assembler = VectorAssembler(inputCols = lista, outputCol='features')

# Prepare data
# df = spark.createDataframe([(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))], ['y', 'features'])

output_data = vec_assembler.transform(df_sample)
# vec_assembler.transform(df_sample.dropna()).show(5)
# counta = output_data.count()
# print(counta)
final_data = output_data.select('y', 'features')

# final_data = final_data.na.drop(subset=["y", "features])


# In[ ]:


from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
# from pyspark.ml import Pipeline
from matplotlib import pyplot
from pyspark.ml.feature import StringIndexer, VectorIndexer

#labelIndexer = StringIndexer(inputCol='y', outputCol='index_y').fit(final_data)
#featureIndexer = VectorIndexer(inputCol='features', outputCol='indexedFeatures', maxCategories=6).fit(final_data)

GBT = GBTClassifier(labelCol='y', featuresCol='features', maxIter=10)

#pipeline = Pipeline(stages=[labelIndexer, featureIndexer, GBT])
# pipeline = Pipeline(stages=[GBT])

model = GBT.fit(final_data)
model.featureImportances
# temp_path = tempfile.mkdtemp()
# whole_path = temp_path + "/gbt_model"
# model.save(whole_path)
# model.featureImportances()


# plot
pyplot.barh(range(len(model.featureImportances)), model.featureImportances)
pyplot.show()


# In[ ]:


important_features = pd.Series(data=model.featureImportances, index=lista)
important_features.sort_values(ascending=False, inplace=True)
plt.figure(figsize=(10, 20))
important_features.nlargest(40).plot(kind='barh')
#important_features.plot(kind='barh')

plt.title('Feature Importances using GBT (maxIter=10)')
#plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)


# In[ ]:


from pyspark.ml.classification import GBTClassificationModel
# sameModel = GBTClassificationModel.load(whole_path)
# sameModel.featureImportances()


# In[ ]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(numTrees=30, maxDepth=5, labelCol='y',featuresCol='features', seed=42)
model_rf = rf.fit(final_data)
model_rf.featureImportances
# plot
pyplot.barh(range(len(model_rf.featureImportances)), model_rf.featureImportances)
pyplot.show()


# In[ ]:


important_features_rf = pd.Series(data=model_rf.featureImportances, index=lista)
important_features_rf.sort_values(ascending=False, inplace=True)
plt.figure(figsize=(10, 20))
important_features_rf.nlargest(40).plot(kind='barh')
#important_features.plot(kind='barh')

plt.title('Feature Importances by Random Forest Classifier')
#plt.barh(range(len(indices)), importances[indices], color='b', align='center')
#plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ## Feature Importance using Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from pyspark.sql.types import DoubleType
rf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=4, max_features=0.5, random_state=2018 )

# rf.fit(df_new2.select(*[col(c).cast("float") for c in df_new2.columns],  df_new_sample.select(df.y.cast("float")))
rf.fit(df_new3,  df_new4)
features = df_new_sample.drop('ACTSYSID','y', 'Role').column.values


# In[ ]:


print((df_new2.count(), len(df_new2.columns)))


# ## Feature Importance using XGBoost
# -- Refer to XGBoost Python scikit-learn API

# In[ ]:


from xgboost import XGBClassifier
from xgboost import gbtree
model = XGBClassifier(booster=gbtree)
# model.fit(df_new_sample.drop('ACTSYSID','y', 'Role').collect(), df_new_sample.select('y').collect()) 
#model.fit(df_new2.select(*[col(c).cast("float") for c in df_new2.columns]),  df_new_sample.select(df.y.cast("float")))
model.fit(df_new3,  df_new4)


# In[ ]:


df_new3 = df_new3.reset_index()
df_new4 = df_new4.reset_index()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(df_new3, df_new4)



###### Cleaning code

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load PySpark
conf = pyspark.SparkConf()
#*80GB of memory to work with on the Spark cluster

sc = pyspark.SparkContext(conf=conf) #Create pySparkContext
sqlContext = SQLContext(sc) #Create sqlSparkContext

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
ccap_DEV_HDFS = dataiku.Dataset("CCAP_DEV_HDFS")
df = dkuspark.get_dataframe(sqlContext, ccap_DEV_HDFS)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def drop_missing(dataframe):
    rows = dataframe.count()
    temp = dataframe.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dataframe.columns]).toPandas()
    columns = [c for c in temp.columns if temp[c][0] == rows ]

    return  dataframe.drop(*list(columns))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df1 = drop_missing(df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def drop_dup(dataframe):
    return dataframe.drop_duplicates()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df1 = drop_dup(df1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def drop_distinct(dataframe):

    temp = dataframe.select([stddev(c).alias(c) for c in dataframe.columns]).toPandas()
    columns = [c for c in temp.columns if temp[c][0] == 0.0 ]

    return dataframe.drop(*list(columns))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df2 = drop_distinct(df1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Drop indentifier colums

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df3 = df2.drop('scan_list_CCAPS_ref_no', 'ip_id','ocif','pri_OCIF_cust_id','fmt_OCIF_id','CKBD_fmt_OCIF_id',
                'MC_CCAPS_prod_pos_nbr', 'corp_nbr', 'cls_doc_file_nbr',
                'MBA_acct_nbr', 'NAS_open_transit_nbr', 'acct_nbr_first4',
               'acct_nbr_last3', 'NAS_fund_acct_nbr')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a SparkSQL dataframe
ccap_DEV_MODELING_df = df3 # For this sample code, simply copy input to output

# Write recipe outputs
ccap_DEV_MODELING = dataiku.Dataset("CCAP_DEV_MODELING")
dkuspark.write_with_schema(ccap_DEV_MODELING, ccap_DEV_MODELING_df)

