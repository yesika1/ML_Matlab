
# coding: utf-8

# # Feature Importance G/B Model (PLOC) - Sheikh

# In[1]:


get_ipython().magic(u'pylab inline')


# #  Importing the libraries

# In[2]:


import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from collections import defaultdict
from itertools import (compress, combinations)

import xgboost as xgb
from xgboost import plot_importance

from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder, 
                                   MinMaxScaler,StandardScaler,Imputer)
from sklearn.feature_selection import chi2
from sklearn.metrics import (mutual_info_score, accuracy_score, r2_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

# dataiku related libraries
import dataiku.core.pandasutils as pdu

# setting pandas options
pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# # Exporting the modelling dataset

# In[3]:


ar = dataiku.Dataset("CCAP_CAD_DAS_PERFORMANCE_joined_filtered")
ar_df = ar.get_dataframe()
df_copy = ar_df


# **This function was added to cluster the G/B data**

# In[4]:


def drop_na_rows_target(dataframe):
    dataframe = dataframe[dataframe['Target'].notnull()]
    dataframe = dataframe.drop(['limits', 'BALANCE', 'actsysid', 'WROF_AMT', 'block', 'status', 'WROF_AMT', 'WROFDATE'], axis = 1)
    return dataframe

ar_df = drop_na_rows_target(ar_df)


# # Let's sneakpeak the data and look at initial statistics

# In[5]:


ar_df.head()


# In[6]:


ar_df.shape


# # Distribution of accounts

# In[7]:


pop_count = ar_df.groupby("Target")["Target"].count()
n = ar_df.shape[0]
pop_prcnt = pop_count/n

plt.bar(x = ["non-default", "default"] , height = pop_prcnt)
plt.xlabel("Class")
plt.ylabel("%")
plt.title("Distribution of accounts")


# Let's do a basic info rundown of the files 

# In[8]:


ar_df.info()


# Let's skim over the numerical variables and summarise them.

# In[9]:


ar_df.describe()


# # Find the proportion of missing values for each column

# In[10]:


# This function will calculate the percentage of missing values across the dataset 
# and list the variables 
def find_missing(data):
    Total = data.isnull().sum().sort_values(ascending = False)
    Percentage = (data.isnull().sum()/data.shape[0]).sort_values(ascending = False)
    
    return pd.concat([pd.DataFrame(Percentage.index), pd.DataFrame(Total.values), 
                      pd.DataFrame(Percentage.values)],
                     axis = 1, keys = ['Var_names','Total','Percent'])


# In[11]:


# The function drops the missing values based on a given percentage threshold
def drop_miss(data, threshold):
    
    missing_stat = find_missing(data)
    varlist_threshold = list(missing_stat['Var_names']                          [missing_stat['Percent'] >= threshold].dropna()[0])
    new_df =  data.drop(labels= varlist_threshold, axis = 1)
    
    return {'new_df' : new_df, 
            'missing_stat' : missing_stat,
            'varlist_threshold': varlist_threshold}

drop_miss_out = drop_miss(ar_df, 0.90)


# Need to inspect the detected missing values to get a better sense before removing them

# In[12]:


varlist_threshold_unique = pd.DataFrame(ar_df[drop_miss_out['varlist_threshold']].nunique())


# In[13]:


ar_df = drop_miss_out['new_df']
ar_df.head()


# In[14]:


ar_df.shape


# # Dropping duplicated rows

# In[15]:


def drop_dup(data):
    return data.drop_duplicates()

ar_df = drop_dup(ar_df)
print((df_copy.shape[0] - ar_df.shape[0]), 
      "rows taken out due to duplication")


# # Dropping some redundant variables
# 
# Went over the business dict to verify this, 
# - Anything that ends with
#     - _ref_no
#     - acct_nbr
#     - _id
#     - _cd
# - Anything that starts with _cls
# - 'ocif' 
# - anything that has 'book'/'nbr' in it, these are mostly related to target variable creation/account identification

# In[16]:


def drop_redundant_vars(data):
    
    var_names = list(data.columns)
    redundant_vars = []
    for col in var_names:

        if col.endswith('_ref_no')        or 'nbr' in col        or col.endswith('_id')        or col.endswith('_cd')        or col.startswith('_cls')        or 'book' in col        or col == 'ocif': 
            redundant_vars.append(col)
            data = data.drop(col, axis = 1)

    print("Total number of columns to be taken out ", len(var_names) - data.shape[1])

    return {'data':data, 'redundant_vars': redundant_vars}

redundant_vars_def = drop_redundant_vars(ar_df)


# Variables which might be worth looking into, 
# - pri_CCAPS_prod_cd
# 
# - CSC_pri_cur_occptn_cd
# 
# - CBR_worst_rtg_cd
# 
# - pri_prov_cd
# 
# - STRATA_last_acty_cd
# 
# - rqst_loan_purpose_cd
# 

# In[17]:


var_exclude = ['pri_CCAPS_prod_cd', 'CSC_pri_cur_occptn_cd', 
               'CBR_worst_rtg_cd', 'pri_prov_cd', 'rqst_loan_purpose_cd']

def redun_vars_filter(data, var_exclude, redundant_vars):
    
    # finding the non-unique catagoricals
    n_unique = data.loc[:,var_exclude].nunique().sort_values(ascending = False)
    varlist_threshold = list(n_unique[(n_unique > 1)].index)

    # filter is against the business defined exclusion after dealing
    # with non-unique variables (unique == 1) 
    filter_vars = list(set(redundant_vars)                              - set(varlist_threshold))
    
    return filter_vars 
        
rdndnt_vars = redun_vars_filter(ar_df, var_exclude, redundant_vars_def['redundant_vars'])
rdndnt_vars[0:5]


# In[18]:


rdndnt_vars


# In[19]:


ar_df = ar_df.drop(rdndnt_vars, axis = 1)


# # Anything ending with _cd or _sid needs to be categorized, so will turn them into strings

# In[20]:



cd_sid_adjust = list(ar_df.columns[[col.endswith("_cd")|col.endswith("_sid") for col in ar_df.columns]])
ar_df[cd_sid_adjust] = ar_df[cd_sid_adjust].astype(str)

# dealing with string NaNs
for col in cd_sid_adjust:
    ar_df.loc[ar_df[col] == 'nan', col] = np.nan


# # Splitting features and target for later use

# In[21]:


var_cols = list(ar_df.columns[ar_df.columns != 'Target'])
X_train, y_train = ar_df[var_cols], ar_df["Target"]


# # Need to deal with the date type columns, they end with _dt
# 
# The steps for this process:
# - First parse out the month and years from the strings
# - Next, turn them into integers 
# - Concatenate them back in and turn them into strings for categorical transformation
# - Excluded day variable since this will create too many uniques

# In[22]:


def ID_date_vars(data):
    dt_tag = []
    var_names = list(data.columns)
    for i in range(len(var_names)):
        var_i = var_names[i].endswith('_dt')
        dt_tag.append(var_i)

    date_vars = list(compress(var_names, dt_tag))
    return data[date_vars]


# In[23]:


date_var_df = ID_date_vars(X_train)
date_vars = list(date_var_df.columns)
date_var_df.head()


# Converting date columns to integers

# In[24]:



def date_converter(dt_col):
    dt_conv = []
    for dt in dt_col:
        dt = str(dt)
        if dt == 'nan':
            dt_conv.append('nan')
        else:
            strp_dt = datetime.datetime.strptime(dt, '%d%b%Y')
            if strp_dt.day in range(10):
                day = '0' + str(strp_dt.day)
            else:
                day = str(strp_dt.day)

            if strp_dt.month in range(10):
                month = '0' + str(strp_dt.month)
            else:
                month = str(strp_dt.month)
            dt_conv.append(int(str(strp_dt.year) + month))

    return dt_conv


# In[25]:


for var_id in date_var_df.columns:
    date_var_df.loc[:,var_id] =         date_converter(list(date_var_df[var_id]))


# In[26]:


date_var_df.head()


# Putting the transformed date variables back

# In[27]:


X_train[date_vars] = date_var_df
X_train[date_vars].head()


# In[28]:


# keeping consistent formatting for date variables
X_train[date_vars] = ar_df[date_vars].astype(str)

# dealing with string NaNs
for col in date_vars:
    X_train.loc[ar_df[col] == 'nan', col] = np.nan


# Using business judgements to manually exclude some variables

# In[29]:



exclude_dates = ['appl_last_upd_dt', 'CCH_acty_dt', 
                'TD_kill_dt', 'MC_xsell3_dt','MC_xsell4_dt', 'crte_dt', 
                'scr_dt', 'adjud_dt', 'data_enter_dt', 'data_comp_dt', 
                'func_area_enter_dt', 'hold_dt', 'inactive_dt',
                 'scan_list_hold_dt','state_enter_dt']


# Listing the variable types

# In[30]:



def var_types(data, date_vars):
    num_var_ids = list(data.select_dtypes(include= [np.number]).columns)
    cat_var_ids = list(data.select_dtypes(include= [object]).columns)
    cat_var_ids = list(set(cat_var_ids) - set(date_vars))
    
    return {'num_vars' : num_var_ids, 'cat_vars' : cat_var_ids}


# In[31]:


num_vars = var_types(X_train, exclude_dates)['num_vars']
cat_vars = var_types(X_train, exclude_dates)['cat_vars']


# # Dropping categorical variables with too many unique factors, default threshold = 100  

# In[32]:



def too_nunique_drop(data, cat_vars, nunique_thresh = 100):
    n_unique = data.loc[:,cat_vars].nunique().sort_values(ascending = False)
    varlist_threshold = list(n_unique[(n_unique > 100) | (n_unique == 0)].index)
    new_df =  data.drop(labels= varlist_threshold, axis = 1)
    
    return {'new_df' : new_df, 'varlist_threshold' : varlist_threshold, 
           'n_unique' : n_unique}


# In[33]:


too_nunique_drop_run =  too_nunique_drop(data = X_train, cat_vars = cat_vars)
varlist_threshold = too_nunique_drop_run['varlist_threshold']
n_unique = too_nunique_drop_run['n_unique']


# In[34]:


varlist_threshold


# These variables may not be a good idea to drop and 
# can be used for further analysis: 
# 
# 1) pri_cur_ocptn_desc: I believe this relates to the borrower's occupation - possible feature engineering 
# 
# 2) BMO_cur_debt_ratio: debt ratio?
# 
# 3) CSC_sats_major_ratio: some sort of a ratio which can be further dig into
# 
# 4) BMO_GDR_TDSR_5_pct: some sort of a code (may be behavioral?)
# 

# In[35]:


df_copy[varlist_threshold].head()


# In[36]:


X_train = too_nunique_drop_run['new_df']


# # Label encoding for all categoricals
# This function produces one hot and label encoder for all object type variables.

# In[44]:


def one_hot_encoder(data, cat_vars):

    def ix_miss(data, var_id):
        ix_miss = list(data[data[var_id].isnull()].index)
        return ix_miss

    def find_non_na(data, var_id):
        na_ix = ix_miss(data, var_id) 
        non_naix = list(set(range(data.shape[0])) - set(na_ix))
        return non_naix
    data = data.reset_index()
    n_unique = data.loc[:,cat_vars].nunique()
    varlist_threshold = list(n_unique[(n_unique > 1)].index)

    cat_df = data.loc[:,varlist_threshold]
    cat_df = cat_df.fillna('MISS').astype(str) # for dealing with NaNs

    label_encoders = {}
    for col in varlist_threshold:
        print("Label Encoding {}".format(col))
        new_est = LabelEncoder()
        ix = find_non_na(cat_df, col)
        cat_df.loc[ix,col] = new_est.fit_transform(cat_df.loc[ix, col].astype('str'))
        label_encoders[col] = new_est

    print("One hot encoding the dataframe...")

    var_ix = list(range(cat_df.shape[1]))
    oh_encode = OneHotEncoder(categorical_features = var_ix,
                              sparse = False,
                              handle_unknown = "ignore")
    oh_encode_fitarray = oh_encode.fit_transform(cat_df)

    print("Creating new dataframe")


    var_labels = []
    for col in varlist_threshold:
        #cls_labels = ar_df[col].unique().tolist()
        var_labels.append([str(col)+'_'+ str(cls_label)                            for cls_label in label_encoders[col].classes_])
    var_labels = [val for sublist in var_labels for val in sublist]

    new_df = pd.DataFrame(oh_encode_fitarray, columns = var_labels)

    # dealing with NaNs
    for new_col in list(new_df.columns):
        if new_col.endswith("_MISS"):
            print("Dropping unnecessary cols:", new_col)
            new_df = new_df.drop(new_col, axis = 1)
            
    return {'ohe_df' : new_df, 'le_df': cat_df}


# In[45]:


X_train_cat = one_hot_encoder(data = X_train, cat_vars = cat_vars)['le_df']


# # Function for imputation
# Default is set to median.

# In[46]:


def impute_data(data, var_ids, strategy):
    
    df = data[var_ids]
    imr = Imputer(missing_values = 'NaN', strategy = strategy, axis = 1).fit(df)
    imputed_data = imr.transform(df.values)
    
    return imputed_data


# # Transforming continuous variables using either normalization or standardization
# In this case, let's use normalization since we can avoid negative value transformations and further ease the process for several feature selection tests.
# 
# Normalization: $$\eta = \frac{x_i - x_{min}}{x_{max} - x_{min}}$$
# 
# Standardization: $$\zeta = \frac{(x_i - \mu)}{\sigma}$$

# In[47]:



def cont_var_transform(data, var_ids, method):
    num_df = data[var_ids]
    num_df_imp = impute_data(num_df, var_ids, 'median')

    if method == 'minmax':
        mm_scaler = MinMaxScaler()
        scaled_data = mm_scaler.fit_transform(num_df_imp)

    if method == 'standardize':
        std_sc = StandardScaler()
        scaled_data = std_sc.fit_transform(num_df_imp)

    scaled_data = pd.DataFrame(scaled_data, columns = var_ids)
    return scaled_data    

X_train_num = cont_var_transform(data = X_train, var_ids= num_vars, method= 'minmax')


# In[48]:


X_train_cat = pd.DataFrame(impute_data(X_train_cat, list(X_train_cat.columns), 'most_frequent'),
                           columns=list(X_train_cat.columns))


# In[49]:


X_train_cat.head()


# # Zero variance treatment

# In[50]:


def zero_var_detect(data):
    
    feature_variance = [np.var(np.array(data[col]),dtype = np.float64)                for col in data.columns]
    feat_list = list(data.columns.values[[feature_variance[ix] > 0                               for ix in range(len(feature_variance))]])
    
    print (len(data.columns.values) - len(feat_list), "features detected with zero variance")
    return(feat_list)


# In[51]:


X_train_num = X_train_num[zero_var_detect(X_train_num)]
X_train_cat = X_train_cat[zero_var_detect(X_train_cat)]


# In[52]:


X_train = pd.concat([X_train_num.reset_index(drop=True), X_train_cat], axis=1)


# In[53]:


list(X_train.columns)


# # Correlation Analysis

# ## Correlation amongst continuous variables

# In[54]:


f, ax = plt.subplots(figsize=(40, 40))
corr = X_train_num.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.title("Correlation Amongst Continuous Variables")


# ## Correlation against Target - Continuous

# In[55]:


f, ax = plt.subplots(figsize=(5, 50))
sns.heatmap(pd.DataFrame([np.absolute(np.corrcoef(y_train, X_train_num[col])[1, 0]) for col in X_train_num.columns],
                         index=list(X_train_num.columns)), ax = ax)
plt.title("Absolute Correlation against Target - Continuous Variables")


# ## $\sqrt{R^{2}}$ against Target - Categorical Variables

# For categoricals, lets create a defined function which calculates the squard root of R Squared based on a linear model 

# In[56]:


cat_vars = var_types(X_train, exclude_dates)['cat_vars']


# In[57]:


def abs_corr(data, y):
    
    abs_corr = []
    lm = linear_model.LinearRegression()

    for col in data.columns:
        x = pd.get_dummies(data[col]).values
        lm.fit(x, y)
        y_pred = lm.predict(x)
        abs_corr.append(np.sqrt(np.max([0,r2_score(y, y_pred)])))
    
    abs_corr = pd.DataFrame(abs_corr, index = list(data.columns))
    
    return abs_corr


# In[58]:


f, ax = plt.subplots(figsize=(5, 50))
sns.heatmap(pd.DataFrame(abs_corr(X_train_cat, y_train),
                         index= list(X_train_cat.columns)), ax = ax)
plt.title("Sqrt R2 - against Target - Categorical Variables")


# # Function for feature selections - ChiSquare and Mutual Information Score

# In[59]:



def uni_variate_tests(X_train,y_train):
    
    print("Calculating chi_test statistics for all features...")
    chi_test = chi2(X_train[list(X_train.columns)], y_train)
    chi2_stat = chi_test[0]
    
    chi2_stat = pd.DataFrame(chi_test[0], index = list(X_train.columns)).    sort_values(by = 0, ascending = False)
    
    p_val = pd.DataFrame(chi_test[1], index = list(X_train.columns)).    sort_values(by = 0, ascending = True)
    
    print("Calculating mi_scores for all features...")
    
    mi_score = []
    for col in X_train.columns:
        mi_score.append(mutual_info_score(X_train[col].values, y_train))
        
    mi_score = pd.DataFrame(mi_score, index = list(X_train.columns)).sort_values(by = 0, ascending = False)          
    
    print('Done!')
    return {'chi2_stat': chi2_stat, 'p_val' : p_val, 'mi_score':mi_score}


# In[60]:


uni_variate_analysis = uni_variate_tests(X_train, y_train)


# In[61]:


chi_res = uni_variate_analysis['chi2_stat'].sort_values(by = 0, ascending = False).head(100)
chi_plt = chi_res.plot(kind = 'barh')
plt.gca().invert_yaxis()
chi_plt.figure.set_size_inches(10, 30)
chi_plt.set_title("Scores - Univariate Chi Square Test")


# Mutual Information Score:
# $$I(X;Y) = \sum_{y}\sum_{x}p(x, y)log(\frac{p(x,y)}{p(x)p(y)})$$

# In[62]:


mi_score_head = uni_variate_analysis['mi_score'].sort_values(by = 0, ascending = False).head(100)
mis_plt = mi_score_head.plot(kind = 'barh')
plt.gca().invert_yaxis()
mis_plt.figure.set_size_inches(10, 30)
mis_plt.set_title("Mutual Information Score")


# # Feature importance calculation using XG Boost
# Model Assumptions:
# 
# - Max depth of each tree is 10 
# 
# - Minimum child weight controls when the tree building should terminate based comparing the sum of the instance weights, set to 1 
# 
# - Objective function is a binary logistic 
# 
# - Subsample ratio of the training instances set to 50%
# 
# - Subsample ratio of columns when constructing each tree set to 30%
# 
# - Subsample ratio of columns for each split, in each level, also set to 30%
# 
# - Positive/negative weight control set to number or reject/number of target
# 
# The parameters for regularizations are default set to zero. 

# In[63]:


def XGB_feat_importance(X_train, y_train, pop_count):
    clf = xgb.XGBClassifier(max_depth = 10, min_child_weight = 1,
                            learning_rate = 0.1, n_estimators = 500,
                            silent = 0, objective = 'reg:logistic',
                            gamma = 0, max_delta_step = 0,
                            subsample = 0.5, colsample_bytree = 0.3,
                            colsample_bylevel = 0.3, reg_alpha = 0,
                            reg_lambda = 0, scale_pos_weight = pop_count[0]/pop_count[1],
                            seed = 1, missing = None)
    clf.fit(X_train, y_train, verbose=True)
    ax = plot_importance(clf)
    fig = ax.figure
    fig.set_size_inches(10, 60)
    plt.show()
    
    return {'clf':clf}


# In[64]:


get_ipython().magic(u'time var_import = XGB_feat_importance(X_train, y_train, pop_count)')


# In[65]:


# Sequential feature selection - still under development
test_size = 0.25 
random_state = 1 
estimator = clone(KNeighborsClassifier(n_neighbors = np.sqrt(X_train.shape[0])))
k_features = 1

def calc_score(X_train, y_train, X_test, y_test, indices):
    estimator.fit(X_train[:, indices], y_train)
    y_pred = estimator.predict(X_test[:, indices])
    score = accuracy_score(y_test, y_pred)

def fit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size,
                                                        random_state = random_state)
    dim = X_train.shape[1]
    indices = range(dim)
    subsets_ = [indices]
    score = calc_score(X_train, y_train, X_test, y_test, indices)
    
    scores_ = [score]
    
    while dim > k_features:
        scores = []
        subsets = []
        
        for p in combinations(indices, r = dim - 1):
            score = calc_score(X_train, y_train, X_test, y_test, p)
            scores.append(score)
            subsets.append(p)
            
        best = np.argmax(scores)
        indices = subsets[best]
        subsets_.append(indices)
        dim -= 1
        
        scores_.append(scores[best])
        print("dimension size reduced to ",  dim)
        
    k_score = scores_[-1]
    
    return {'indices': indices,'scores_': scores_, 'k_score':k_score}

def transform(X):
    return X[:, indices]


# In[66]:


num_dat_transform = X_train_num


# In[67]:


# principal feature analysis - still under development
"""
Step 1: Compute the sample covariance matrix, or use the true covariance matrix if it is
available. In some cases it is preferred to use the correlation matrix instead of the
covariance matrix.

Step 2: Compute the Principal components and eigenvalues of the
Covariance/Correlation matrix.

Step 3: Choose the subspace dimension q and construct the matrix Aq from A. This can
be chosen by deciding how much of the variability of the data is desired to be
retained. The retained variability is the ratio between the sum of the first q
eigenvalues and the sum of all eigenvalues.

Step 4: Cluster the vectors V1, V2,...,Vn ∈ ℜ(q) to p ≥ q clusters using K-Means
algorithm. The distance measure used for the K-Means algorithm is the
Euclidean distance. Choosing p greater than q is usually necessary if the same
variability as the PCA is desired (usually 1-5 additional dimensions are needed)

Step 5: For each cluster, find the corresponding vector Vi which is closest to the mean of
the cluster. Choose the corresponding feature, xi , as a principal feature. This
step will yield the choice of p features. The reason for choosing the vector
nearest to the mean is twofold. This feature can be thought of as the central
feature of that cluster- the one most dominant in it, and which holds the least
redundant information of features in other clusters. Thus it satisfies both of the
properties we wanted to achieve- large “spread” in the lower dimensional space,
and good representation of the original data.

"""
q = num_dat_transform.shape[1]
pca = PCA(n_components = q).fit(num_dat_transform)
A_q = pca.components_.T

kmeans = KMeans(n_clusters= 50).fit(A_q)

clusters = kmeans.predict(A_q)

clusters_centers = kmeans.cluster_centers_

dists = defaultdict(list)
for i, c in enumerate(clusters):
    dist = euclidean_distances([A_q[i, :]], [clusters_centers[c, :]])[0][0]
    dists[c].append((i, dist))

indices = [sorted(f, key = lambda x: x[1])[0][0] for f in dists.values()]


# In[68]:


doc = df_copy["pri_cur_ocptn_desc"]
doc = doc.fillna("NOT AVAILABLE")
# some feature engineering on the occupation column
def term2doc_df(docs, doc_id, **kwargs):

    
    init_vectorizer = CountVectorizer(**kwargs)
    dtm_fit = vectorizer.fit_transform(docs)
    
    #create dtm dataFrame
    df = pd.DataFrame(dtm_fit.toarray(), 
                      index = doc_id)
    
    df.columns = vectorizer.get_feature_names()

    return df

term2doc_df(doc,  doc)

