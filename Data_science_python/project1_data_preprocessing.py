#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Hi Yesika,
# 
# I have checked you work and left comments in such cells. Cells are of two types:
# 
# <div class="alert alert-block alert-danger">
# <p> <strong> A red colored cell </strong> indicates that you need to improve or adjust part of the project above. </p>
# </div>
# <div class="alert alert-block alert-info">
# <p> <strong> A blue colored cell </strong> indicates that no improvements are needed in the cells above. May include some suggestions and recommendations.</p>
# </div>
# 
# Hope it all will be clear to you :)
# 
# You did a great job 😀. You understand what you are doing and why, also can make right conclusions - which is great 👍.
# 
# Thank you for keeping your notebook clean and structured, with clear explanations :)
# 
# Project is accepted :)
# 
# *Good luck!*
# 
# ------------

# ## Analyzing borrowers’ risk of defaulting
# 
# Your project is to prepare a report for a bank’s loan division. You’ll need to find out if a customer’s marital status and number of children has an impact on whether they will default on a loan. The bank already has some data on customers’ credit worthiness.
# 
# Your report will be considered when building a **credit scoring** of a potential customer. A ** credit scoring ** is used to evaluate the ability of a potential borrower to repay their loan.
# 
# The data may contain artifacts, or values that don't correspond to reality—for instance, a negative number of days employed. You need to describe the possible reasons such data may have turned up and process it. Data analysis is about more than just running calculations e.g. there’s also looking for outliers, artifacts, and errors in the data too.

# ### Data dictionary
# - children : the number of children in the family
# - days_employed: how long the customer has worked
# - dob_years: the customer’s age
# - education: the customer’s education level
# - education_id: identifier for the customer’s education
# - family_status: the customer’s marital status
# - family_status_id: identifier for the customer’s marital status
# - gender: the customer’s gender
# - income_type: the customer’s income type
# - debt: whether the client has ever defaulted on a loan
# - total_income: annual income
# - purpose: reason for taking out a loan

# In[1]:


import pandas as pd  # process dataframes
import numpy as np # process arrays
import matplotlib.pyplot as plt #plotting
import seaborn as sns # plotting
plt.style.use('ggplot')

from nltk.stem import SnowballStemmer # Natural Language Processing


# ### Step 1. Open the data file and have a look at the general information. 

# In[2]:


## Reading dataset
df = pd.read_csv('/datasets/credit_scoring_eng.csv')

# General information including the numbers of objects within each column
print(df.info())


# In[3]:


## update features names
#[col for col in df.columns]
#[col.lower().replace(' ','_') for col in df.columns]


# In[4]:


# Displaying first two rows
display(df.head(2))


# In[5]:


# Summarizing only numerical columns
display(df.describe(include = [np.number]))


# In[6]:


# Summarizing only categorical columns
display(df.describe(include = ['O']))


# In[7]:


## calculating the probability of default (debt =1)
df['debt'].mean()


# **We have 8% default rate**

# In[8]:


## calculating the probability of default (debt =1)
df.pivot_table(index = 'gender',values = ['debt'], aggfunc = ['count','sum','mean'])
# using group by
df.groupby(['gender'])['debt'].agg(['count','sum','mean'])


# In[9]:


df.groupby(['gender','debt'])['debt'].agg(['count'])


# In[10]:


## highlight if probability of default greater than the general value (8%)

(df.groupby(['gender'])['debt']
.agg(['count', 'mean'])
.sort_values('count',ascending=False)
.style
.format({
    'mean': '{:,.1%}'.format,
})
.applymap(
     lambda x: 'background-color : limegreen' if x>0.08 else '', 
     subset=['mean']))


# ### Conclusion

# At a first glance we can observe that:
# - There are 21525 observations and 12 features in the credit scoring dataset.
# - There are 4 categorial features, and 8 numeric features (type int and float).
# - The following features have missing values: days_employed, and total_income.
# 
# Some data will require preprocessing, for instance:
# - Some features might present collinearity. For example, education and education_id.
# - Some data types might need to be replace it. For example, days_employed, as it should be an integer number.
# - Al least 25% of values in days_employed have negative values. It means nothing. Just rubbish.
# - The minimum value for children is -1, which could be a dead member, but we are going to assume that this data has errors - human factor.
# - The maximum value for children is 20, which it is also incorrect.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Great :)
# 
# I won't agree with you on these points for now:
# 
# <blockquote>
#     <ul>
#         <li>Al least 25% of values in days_employed have negative values. It means nothing. Just rubbish.</li>
#         <li>The maximum value for children is 20, which it is also incorrect.</li>
#     </ul>
# </blockquote>
# 
# We'll see what you will decide further :)
# 
# **UPD**. It is all well explained and processed by you further on step 2 👍
# 
# ------------

# ### Step 2. Data preprocessing
# 
# - Identify and fill in missing values.
# - Replace the real number data type with the integer type.
# - Delete duplicate data.
# - Categorize the data.

# ### Processing missing values
# 
# The Problem with leaving nulls in the dataset or redefining them as 'NA' is that it does not solve the problem and we cannot use them in our analysis. Each time when we need to use features with null values if we just discard those null observations and thus reduce size of the sample as well as dataset in general. By doing so we include bias in the reduced dataset, make it different from the original one. Thus we cannot combine conclusions derived from the reduced dataset with those that were obtained from the dataset with null observations, as these conclusions are based on different data.
# 
# I believe to replace NaNs with special value/symbol and keep it in mind during analysis is a good approach. Under “keep it in mind” I mean, that you don’t evaluate mean/median through such values, just skipping them. It introduce less bias, than filling such values with mean/median inside corresponding group. Also if such NaNs is completely random, there won’t be any bias in data. So I consider this approach is the best for your goals.

# In[11]:


# First, we are going to identify null values using the isnull() method, which checks for missing values in a column. 
print(df.isnull().sum())


# In[12]:


## We are going to start imputating values for the features: days_employed, and total_income. 
#Let’s take a look at the rows that contain NaN.
display(df[df['total_income'].isnull()].head())


# -  It seems that the rows with missing values for days_employed, also have missing values for total_income. We are going to assume that these NaNs occurred when a customer did not have a job. Therefore, we will fill NaNs with the number 0 for these two numeric features.
# - Also, as the instructor suggested, zero is a better option, because it’s very important column for banks and it’s extreme unlikely that banks leave this column empty for another reason.

# In[13]:


# Replacing NaNs values
df['total_income'] = df['total_income'].fillna(0)
df['days_employed'] = df['days_employed'].fillna(0)

# Verifying imputation
print(df.isnull().count())


# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Good :)
# 
# ------------

# In[14]:


## Identifying  missing values for categorical values
print(df['gender'].value_counts())


# In[15]:


## There is one observation with a missing value, having a level of "XNA".
# Let's observe this row 
display(df [ df['gender']== 'XNA'])


# - As there is only one observation as missing value for the gender feature, which represent less than the 1% of the data. We will drop this value.

# In[16]:


length_before = len(df)
df = df.drop([10701],axis=0) # dropping index 10701
length_after = len(df)

print('length before: {} and after dropping a row: {}'.format(length_before,length_after))


# ### Conclusion

# - Missing values for the numeric features: **days_employed, and total_income** were identified using the using the isnull() method, which checks for missing values in a column. 
# 
# A possible reason for these missing values was that these missing values occurred when a customer did not have a job, as the rows with missing values for days_employed, also have missing values for total_income. 
# 
# In order to avoid losing rows with important data, we replaced the NaN value in the columns with zeros, using the fillna() method (filling with N/A), where the argument is a substitute for the missing values. 
# 
# - Missing values for the categorical feature: **gender** was identified using the value_counts() method which count the unique values whithin the feature. 
# 
# As there is only one observation as missing value for the gender feature, the possible reason for this behavior was a data entry error. 
# 
# This missing value represents less than the 1% of the data. Threfore, we dropped this value from the dataset using the drop() method.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Great :)
# 
# ------------

# ### Data type replacement
# 
# **Note from instructor**: Real number is float data type. If you feature has integer nature, it’s better to store it as int. For instance, if your feature is the number of floor, it’s strange to store it as a float, because you expect it to be integer. Also float numbers take more memory than int.

# There are two variables with float type, when we obbserve the info table:
#     - days_employed
#     - income
# And only days_employed makes sense to be changed to integer type. As income, in financial terms, will be a float number.

# In[17]:


# Repalcing data type
df['days_employed'] =df['days_employed'].astype('int')

# Verifying data type
print(df.info())


# ### Conclusion

# Replace the real number data type with the integer type.
# —which method you used to change the data type and why;
# —which dictionaries you've selected for this data set and why.
# 
# 
# There are two variables with float type, when we obbserve the info table:
#     - days_employed
#     - income
# And only days_employed makes sense to be changed to the integer type. As income, in financial terms, will be a float number.
# 
# The data type of days_employed was transformed using the .astype('int') method.
# 

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Great :)
# 
# Also, `debt` datatype should be changed to boolean, as it is not a quantitative feature.
# 
# ------------

# ### Processing duplicates
# - Each row corresponds to a customer.

# In[18]:


## first lets put the content of the education feature as lowercase:
# if we don't clean the data in this column, the .duplicated() method won't consider "GRADUATE DEGREE" and "graduate degree"
print('Before:')
print(df['education'].value_counts())
print('------')
df['education'] = df['education'].str.lower()
print('After:')
print(df['education'].value_counts())


# In[19]:


# Searching for duplicate data
print('Duplicated rows:',df.duplicated().sum())
#print(df[‘col’].duplicated().sum())

# Reviewing some duplicates:
print('Reviewing some duplicates:')
display(df[df.duplicated()].head(10))

# Deleting duplicates
df.drop_duplicates(inplace=True)

# verifying duplicated rows
print('Veryfing duplicated rows:',df.duplicated().sum())

# Reseting the index number
df = df.dropna().reset_index(drop=True)


# ### Conclusion

# 54 duplicated rows were identified using the df.duplicated() and sum() method. Where the duplicated() method returns Series with the value True if there are duplicates, False if there aren’t. When we partner it up it with sum(), it returns the number of duplicates.
# 
# After identified the duplicates, I reviewed the top 10 rows and no pattern was identified. Therefore, we attribute the dupplicate to entry data repetitions, human factor.
# 
# Finally, the duplicates were removed from the dataset using the drop_duplicates() method. And we reseted the index with the dropna().reset_index(drop=True) method. Because when we call the drop_duplicates() method, the rows containing repetitions are deleted including their indices.
# 
# Further analysis of the feature content will be done in the Categorizing Data step.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Great :)
# 
# ------------

# ### Categorizing Data
# 
# **Notes from instructor**: The data may contain artifacts, or values that don't correspond to reality—for instance, a negative number of days employed. This kind of thing happens when you're working with real data. You need to describe the possible reasons such data may have turned up and process it.
# 
# - For instance, for **days_employed**:
# If you’ll plot histogram of such data and see that distribution of negative values is different out of positive values, you can assume, that something is wrong with them. If you want to check the problem is only with a sign, you can take absolute value and plot the same histogram again. If now negative values look like fine, their distribution is coincided with distribution of positive values, you can suggest that the problem really was only with sign.
# As you can see I nowhere mentioned type of distribution, thus it’s not necessary to be normal.
# About days_employed - yes, you can leave it as it is, because you don’t need it in your final analysis.
# 
# - Now, for the **children** feature:
# You can’t say for 100% that 20 children is a outliers. Also it doesn’t matter. You will analyze categorized children feature. So rows with children is equal to 20 will be putted to the category, for instance, “more than 5 children”.
# How do we interpret the -1 value? How much rows this -1 children? If just several, you can drop them. If a lot, just make special category for that. For instance, “unknown”. If you don’t have a description about ‘-1’ values, it’s better to think that it’s a bug in data. Using “unk” values for it is a good choice.
# 
# 
# - for the variable **purpose**:
# Let’s go step by step:
# If you use only pos=“n” in method lemmatize, then only noun words will be lemmatized. In the exerciser there was a task how to deal with it. You should check the type of each word and use it in the argument pos.
# Firstly, you should to lemmatize each purpose and store it in a new column
# Secondly, you should write a function which you will use next in the .apply method.
# You need to apply your function from step 3 to the result of step 2 and get new categorized column.
# About this magic function. You dealt with very similar function in the exerciser, when you categorized age of people. This function should get row and return the category. It’s very similar to your loop, but the function should return category.
# 
# - Just print variable pos_tag and you will understand why this happend.
# tag_dict = {"J": wordnet.ADJ,
#             "N": wordnet.NOUN,
#             "V": wordnet.VERB,
#             "R": wordnet.ADV}
#  
# tag = nltk.pos_tag([word])[0][1][0].upper()
#  
# new_word = lemmatizer.lemmatize(word, tag_dict.get(tag, wordnet.NOUN))

# In[20]:


## Starting with the days_employed feature
# Density Plot and Histogram of days_employed

sns.distplot(df['days_employed'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Distribution of days_employed')
plt.xlabel('Days employed')


# In[21]:


# As we can see the from the plot histogram that the distribution of negative values 
# is different out of positive values, we can assume, that something is wrong with these values.

# Let's check if the problem is only with a sign.

# Density Plot and Histogram of the absolute value of days_employed

sns.distplot(abs(df['days_employed']), hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# Add labels
plt.title('Distribution of Absolute value of days_employed')
plt.xlabel('bsolute value of Days employed')

# After taking the absolute value and plot the same histogram again. 
# We cannot conclude that the problem really is only with the sign,
# as their distribution is skwew and it is hard to assume that coincided with distribution of positive values.
# Since this variable will be not used for further analysis in the following steps. We will not procede to do a further analysis.


# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# In `days_employed` extremely large values (about 1000 years) belong to observetions that are either `unemployed` or `retiree` (`income_type` categories). So, there seems to be a pattern :)
# 
# ------------

# In[22]:


## Analyzing the variable children

# Density Plot and Histogram of children
import seaborn as sns
sns.distplot(df['children'], hist=True, kde=True,
             bins=int(180/5), color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# Add labels
plt.title('Distribution of children')
plt.xlabel('Number of children')


# In[23]:


# for the variable children most of the values are less than 5.

# Let's find out how many observations where captured as -1
print("There are: {} cases with value -1 for the feature children".format(
    len(df[df['children'] == -1])))


# In[24]:


## categorizing the variable children 

def children_group(children): 
    """ The function returns the children group according to children quantity, 
        by using the following rules: 
        - 'children', with value <= -1  :'unknown', 
        - children', with value = 0  :'without children', 
        - 'children', with value over  5  and up to 64 : 'more than 5 children' 
        - for all other cases : '1 to 4 children'
    """ 
    try:
        if children <= -1: return 'unknown' 
        elif children == 0:  return 'without children' 
        elif children >= 5:  return 'more than 5 children' 
        else: return '1 to 4 children'
    except:
        print('Check that the values are numeric') 
#print(children_group(-1)) # unknown


# Creating a separate column for children categories
df['children_group'] = df['children'].apply(children_group) 
display(df.head(3))

#Now let's get data on children groups with the value_counts() method:
display(df['children_group'].value_counts())


# In[25]:


## Identifying categorical values for the variable purpose

print(df['purpose'].nunique())
print('------')
print(df['purpose'].value_counts())


# In[26]:


# As we can see, there are multiple categories that can be rearranged. For instance the purpuses: 
# wedding ceremony, having a wedding, and to have a wedding can be group together as 'wedding expenses'

# We are going to use NLTK package for this purpose.
eng_stemmer = SnowballStemmer('english')

# first, we identify the common keywords and their stems to identify duplicated values later.
words = ['wedding', 'estate', 'housing', 'property', 'house', 'car', 'education', 'educated', 'university']
stem_dict = {}
for word in words:
    print('Word: {}, Stem: {}'.format(word, eng_stemmer.stem(word)))
    stem_dict[eng_stemmer.stem(word)] = word
print('-------')
print ('initial dictionary:',stem_dict)

# adjusting the dictionary to the new levels for the variable purpose
stem_dict['estat'] = 'commercial housing'
stem_dict['hous'] = 'private housing'
stem_dict['properti'] = 'private housing'
stem_dict['educ'] = 'education'
stem_dict['univers'] = 'education'
print('-------')
print ('final dictionary:', stem_dict)


# In[27]:


# Second, we define a function to match the stem_dict with every value in the column purpose
def purpose_stem(row, stemmed_dict):
    ''' 
    This function identify the stem of a word given a string (sentence).
    '''
    # first lets find the stem splitting the string by space
    try: 
        stemmed = [eng_stemmer.stem(word) for word in row.split(' ')]
        # print(stemmed)

        # let's evaluate the list of stemmed found previously
        # if 'wed' in stemmed: return 'wedding'
        for key,value in stemmed_dict.items():
            if key in stemmed: 
                return value
    except:
        print('Check that the values are strings')
        
#purpose_stem(df['purpose'][0])
#df['purpose'].apply(purpose_stem) 

purpose_stem(df['purpose'][0], stem_dict)
df['purpose_category'] = df['purpose'].apply(purpose_stem, stemmed_dict =stem_dict)

#checking the output
print("Checking if the categorization works:")
display(df[['purpose','purpose_category']].head(10))

## Checking the frequency
print("frequency of new levels: \n", df['purpose_category'].value_counts())


# ### Conclusion

# Some features were categorized to facilitate the analysis. For isntance, the data that might contain artifacts, or values that do not correspond to reality and features with multiple levels.
# 
# - Starting with the days employed varible, this variable had multiple negative values. After analyzing the distribution of negative values before and after calculating the absolute value, we cannot conclude that the problem depends only with the sign of the value. Moreover, this variable will be not used for further analysis in the following steps. We will not procede to do a further preprocessing.
# 
# - For the children feature, after analyzing its distribution and noticing that most of the values are less than 5, I procedeed to categorize the children, given this number and also to reagroup the values with -1 as unknown, as we do not want to drop this observations (47 rows). For this a function was defined to classify the 3 new categories, and then applied to the series using the apply function.
# 
# - Finally, for the purpose feature, initially we had 38 different responses that were reagroup into 5 new levels using the stem of the words. For this purpose, we starting identifying the key words and saving the stem as a dictionary. Then, some of the stems were reagrouped in order to create more concise levels for this feature. After that a function that identify the stem of a word given a string (sentence) was created  and finally applied to the dataframe using the apply function.
# 

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Excellent :)
# 
# 'private housing' and 'commercial housing' categories can be grouped into one category 'real_estate' in this project :)
# 
# You have studied almost all the columns in the dataset and indetified almost all the problems. The only thing that you've missed is 0-year old clients 🤓
# 
# ------------

# ### Step 3. Answer these questions

# **- Is there a relation between having kids and repaying a loan on time?**
# 
# if debt = 0, it can be considered as “on-time repayment of loans”, if debt = 1, then it means they have defaulted before.

# In[28]:


display(df['children_group'].value_counts())


# In[29]:


display(df['debt'].value_counts())


# In[30]:


## Total observations among groups
print(df.groupby(['children_group','debt'])['debt'].count())


# In[31]:


# Percentage of obervations among groups 
round(df.groupby(['children_group','debt'])['debt'].count()/df.groupby(['debt' ])['children_group'].count(),3)


# In[32]:


# percentage of debt per category (where every category is 100%)
round(df.groupby(['children_group','debt'])['debt'].count()/df.groupby(['children_group'])['debt'].count(),2)


# In[33]:


df.groupby(['children_group','debt']).size().unstack().plot(kind='bar',stacked=True)
plt.legend()
plt.title("Debt by Children group")
plt.show()


# ### Conclusion

# - If we look at the children categories distribution, the majority of the data corresponds to without children and 1 to 4 children, being around 90% of the data, while the other remaining 10% of the data is distributed among more than 5 children and unknown.
# 
# - The distribution of debt among the different categories is unbalanced (non evenly distributed) and unstable among categories, having a ratio around 92% of observations with debt against 8% observations without debt while the unknown group has a ratio of 98% without debt.
# 
# - We can infer that the different family status do affect on-time repayment of the loan due that the debt is not constant in its distribution among the status and we may need to look further into the unknown group or remove these observations.

# **- Is there a relation between marital status and repaying a loan on time?**
# 

# In[34]:


## Total observations among groups
print(df.groupby(['family_status','debt'])['debt'].count())


# In[35]:


# Percentage of obervations among groups 
round(df.groupby(['family_status','debt'])['debt'].count()/df.groupby(['debt' ])['family_status'].count(),2)


# In[36]:


# percentage of debt per category (where every category is 100%)
round(df.groupby(['family_status','debt'])['debt'].count()/df.groupby(['family_status'])['debt'].count(),2)


# In[37]:


df.groupby(['family_status','debt']).size().unstack().plot(kind='bar',stacked=True)
plt.legend()
plt.title("Debt by family status ")
plt.show()


# ### Conclusion

# - If we look at the family status distribution, the majority of the data corresponds to married status, being around 60% of the data, while the other half of the data is distributed among the civil partnership (~ 20%), following by unmarried (~ 10%), widow (~ 5%) and divorced (~ 5%).
# 
# - The distribution of debt among the different categories is unbalanced (non evenly distributed) and constant among categories, having a ratio around 92% of observations with debt against 8% observations without debt.
# 
# - We can infer that in general the different family status do not affect on-time repayment of the loan due that the debt has a constant distribution tendency. However, unmarried presents the higher probability of default being 10%, and widow and divorced presents the lower probability of default being 7%.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Great :)
# 
# ------------

# **- Is there a relation between income level and repaying a loan on time?**

# In[38]:


df[['income_type','total_income']].head(5)


# In[39]:


plt.figure(figsize=(10,6))
df[df['debt']==0]['total_income'].hist(bins=35,color='blue', alpha=0.6,
                                              label='debt = 0')
df[df['debt']==1]['total_income'].hist(bins=35,color='red', alpha=0.6,
                                              label='debt= 1')
plt.legend()
plt.xlabel("total_income")
plt.title("Debt by total income")


# ### Conclusion

# - From the plot we can observe that the total income is skew to the right with outliers as usually is due its nature.
# 
# - We can see that the probability of default presents the same distribution and it is unbalanced as in the previous cases. 
# 
# - We can infer that in general the total income affects the on-time repayment of the loan due that the debt tends to be higher for total income is less than 50000.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Here it is also better to calculate the probability of failing on debt for each group :)
# 
# ------------

# In[55]:


# Another approach
df.plot(kind='scatter',x='total_income',y='debt',alpha=0.1)


# In[49]:


# correlation, metric that find the connection between two variables
df[['total_income', 'debt']].corr()


# In[57]:


df.groupby(['total_income'])['debt'].median().plot()


# **As we can see, there is a lot of noise, we cannot interpreted the distribution. We are going to analyze the total_income feature and categorize it into groups.**

# In[40]:


# Analyzing total income and categorizing it into subgroups.
df['total_income'].hist()


# In[41]:


## dividing the real variable in 5 bins
print(pd.cut(df['total_income'],5).value_counts())
print('-------')

## The data is not distributed in equal parts among bins, lets distribuite the data manually in six bins
# To prevent noise and wrong rules.
print(pd.cut(df['total_income'],[0,20000,30000,40000,50000,np.inf]).value_counts())


# In[42]:


# creating a new feature 
df['total_income_group'] = pd.cut(df['total_income'],[0,20000,30000,40000,np.inf])
df['total_income_group'].head(2)
#df['amount_group'] = pd.cut(df['credit_amount'],[0,1000,2000,3000,5000,np.inf])


# In[43]:


df['total_income_group'].value_counts().plot(kind='bar')


# In[44]:


df.groupby(['total_income_group'])['debt'].mean().plot()


# **The bigger default is found in the group with income between 20K and 30K.**

# **- How do different loan purposes affect on-time repayment of the loan?**

# In[45]:


## Total observations among groups
print(df.groupby(['purpose_category','debt'])['debt'].count())


# In[46]:


# Percentage of obervations among groups 
round(df.groupby(['purpose_category','debt'])['debt'].count()/df.groupby(['debt' ])['purpose_category'].count(),2)


# In[47]:


# percentage of debt per category (where every category is 100%)
round(df.groupby(['purpose_category','debt'])['debt'].count()/df.groupby(['purpose_category'])['debt'].count(),2)


# In[48]:


df.groupby(['purpose_category','debt']).size().unstack().plot(kind='bar',stacked=True)
plt.legend()
plt.title("Debt by loan purpose_category ")
plt.show()


# ### Conclusion

# - If we look at the purpose distribution, 30% of the data corresponds to private housing while wedding only has 11% of the data. The other purposes have around 20% of data each.
# 
# - The distribution of debt among the different categories is unbalanced (non evenly distributed) and constant among categories, having a ratio around 92% observations with debt against 8% observations without debt.
# 
# - We can infer that in general the different purpose categories do not affect on-time repayment of the loan due that the debt has a constant distribution tendency. However, private housing presents the lower probability of default being 7%.

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Good :)
# 
# ------------

# ### Step 4. General conclusion

# - The original dataset had 21525 observations and 12 features in the credit scoring dataset.
# 
# - Variables that were used in the relationship analysis with the debt feature were transformed if required.
# 
# - After removing duplicates and dropping/imputating missing values and transforming (adding new categories) to some features, we ended with 21453 observations and two new features:children_group, and purpose_category.
# 
# - The distribution for the debt feature is unbalanced, were most of the cases fall into not default as expected.
# 
# - The debt feature was compared against multiple variables an the probability of default percentage was almost the same for all the cases (92% of prob of default)
# 
# - As a recommendation, further analysis might need to be done if we want to use these features for modeling the probability of default or  building a credit scoring. For instance, balancing the debt feature might be required. 

# <div class="alert alert-block alert-info">
# <h2> Comments </h2>
# </div>
# 
# Very good :)
# 
# ------------

# ### Project Readiness Checklist
# 
# Put 'x' in the completed points. Then press Shift + Enter.

# - [x]  file open;
# - [ ]  file examined;
# - [ ]  missing values defined;
# - [ ]  missing values are filled;
# - [ ]  an explanation of which missing value types were detected;
# - [ ]  explanation for the possible causes of missing values;
# - [ ]  an explanation of how the blanks are filled;
# - [ ]  replaced the real data type with an integer;
# - [ ]  an explanation of which method is used to change the data type and why;
# - [ ]  duplicates deleted;
# - [ ]  an explanation of which method is used to find and remove duplicates;
# - [ ]  description of the possible reasons for the appearance of duplicates in the data;
# - [ ]  data is categorized;
# - [ ]  an explanation of the principle of data categorization;
# - [ ]  an answer to the question "Is there a relation between having kids and repaying a loan on time?";
# - [ ]  an answer to the question " Is there a relation between marital status and repaying a loan on time?";
# - [ ]   an answer to the question " Is there a relation between income level and repaying a loan on time?";
# - [ ]  an answer to the question " How do different loan purposes affect on-time repayment of the loan?"
# - [ ]  conclusions are present on each stage;
# - [ ]  a general conclusion is made.
