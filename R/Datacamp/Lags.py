
# coding: utf-8

# In[ ]:


# You have to first initialize the outer list with lists 
# before adding items

matrix1 = [[]]
type(matrix1)
cols = 3
rows = 4
matrix1 = [ [0 for x in range(cols) ] for y in range (rows) ]
print(matrix1)


# In[ ]:


[ 'p'  for y in range (rows) ]


# In[ ]:


['g' for x in range(cols)] 


# In[ ]:


[ ['g' for x in range(cols) ] for y in range (rows) ]


# In[ ]:


#create the identical matrix

matrix1 = [[]]
cols = 3
rows = 4
for y in range (rows):
    for x in range(cols):
        if y == x: 
            value = 1
            matrix1.append(value)
        else: 
            value= 0
            matrix1.append(value)
print(matrix1)


# In[ ]:


def identity(n):
    m=[[0 for j in range(n)] for i in range(n)]
    for i in range(0,n):
        m[i][i] = 1
    return m

identity(2)


# In[ ]:


def identity2(n):
    identity = [[0]*i + [1] + [0]*(n-i-1) for i in range(n) ]
    return identity
identity2(2)


# In[ ]:


index= []
def add_to_index(index,keyword,url):
    for a in index:
          if a[0]==keyword:
              a[1].append(url)
              return
    index.append([keyword, [url]])
    
add_to_index(index,'udacity','http://udacity.com')
add_to_index(index,'computing','http://acm.org')
add_to_index(index,'udacity','http://npr.org')
print (index)


# In[ ]:


index = []
def add_to_index(index,keyword,url):
    for value in index:
        if value[0] == keyword:
            value[1].append(url)
        else:
            index.append([keyword,[url]])

add_to_index(index,'udacity','http://udacity.com')
add_to_index(index,'computing','http://acm.org')
add_to_index(index,'udacity','http://npr.org')
print (index)


# In[ ]:


index= []
def add_to_index(index,keyword,url):
    for a in index:
        if a[0]==keyword:
            a[1].append(url)
              #return
        else:
            index.append([keyword, [url]])
    
add_to_index(index,'udacity','http://udacity.com')
add_to_index(index,'computing','http://acm.org')
add_to_index(index,'udacity','http://npr.org')
print (index)


# In[ ]:


list = 'ayer fue ayer'
list.split()


# In[ ]:


list1 =",!-"
x= list(list1)
x


# In[ ]:


import os 
def rename_files():
    path = 'home/fff'
    file_list = os.listdir(path)
    saved_path = os.getcwd()
    os.chdir(path)
    
    for file in file_list:
        print('old name: '+file_name)
        os.rename(file_name, file_name.translate(None,'0123456789'))
    os.chdir(saved_path)

rename_files()


# In[ ]:


names =  input('Enter names, separated by commas: ').split(',').title()


# In[ ]:


assignments =  input('Enter assignment counts, separated by commas: ').split(',')


# In[ ]:


grades = input("Enter grades separated by commas: ").split(",")


# In[ ]:


for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))


# In[ ]:


names = input("Enter names separated by commas: ").title().split(",")
assignments = input("Enter assignment counts separated by commas: ").split(",")
grades = input("Enter grades separated by commas: ").split(",")

message = "Hi {},\n\nThis is a reminder that you have {} assignments left to submit before you can graduate. You're current grade is {} and can increase to {} if you submit all assignments before the due date.\n\n"

for name, assignment, grade in zip(names, assignments, grades):
    print(message.format(name, assignment, grade, int(grade) + int(assignment)*2))


# In[ ]:


def party_planner(cookies, people):
    leftovers = None
    num_each = None
    # TODO: Add a try-except block here to
    #       make sure no ZeroDivisionError occurs.
    while True:
        try: 
            num_each = cookies // people
            leftovers = cookies % people
            
            break
        except ZeroDivisionError:
            print('Number not valid')
            #break
        except TypeError:
            print('Number not valid')
            
        return(num_each, leftovers)
# The main code block is below; do not edit this
lets_party = 'y'
while lets_party == 'y':

    cookies = int(input("How many cookies are you baking? "))
    people = int(input("How many people are attending? "))

    cookies_each, leftovers = party_planner(cookies, people)

    if cookies_each:  # if cookies_each is not None
        message = "\nLet's party! We'll have {} people attending, they'll each get to eat {} cookies, and we'll have {} left over."
        print(message.format(people, cookies_each, leftovers))

    lets_party = input("\nWould you like to party more? (y or n) ")


# In[ ]:


l=100
def pr_len():
    return(l)
print(pr_len())
print(l)


# In[ ]:


l=100
def pr_len():
    l =20
    return(l)
print(pr_len())
print(l)


# In[ ]:


l=100
def pr_len():
    l =20
    l = l+10
    return(l)
print(pr_len())
print(l)


# In[ ]:


l=100
def pr_len():
    global l
    l = l+10
    return(l)
print(pr_len())
print(l)


# In[ ]:


x = [1,2,3]
y = list(x)
z = x[:]
w = x
print(x)
y = y.append(4)
print(x)
z[1] = 6
print(x)
w[0] =2
print(x)


# In[ ]:


t = x + [9,8]
print(x)
print(t)


# In[ ]:


x= x +[4,5]
print(x)


# In[ ]:


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
print(type(areas))
print(areas)

# Use append twice to add poolhouse and garage size
areas.append(24.5)
print(type(areas))
print(areas)
#areas = areas.append(15.45)


# In[ ]:


x= [1,2,3]
y= [2,4,6]
import numpy as np
np_x = np.array(x)
np_y = np.array(y)
print(x,y)
print(np_x,np_y)


# In[ ]:


np_df1 = np.column_stack((x,y))
print(np_df1)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()


# In[ ]:


plt.plot(np_x,np_y)
plt.show()


# In[1]:


# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


# In[4]:


def echo(n,word1):
    """Return the inner_echo function."""
    echo_word = word1 * n
    return echo_word
twice =echo(2,'hello')
print(twice)


# In[43]:


import pandas as pd
df =pd.read_csv('Dummy.csv')


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


#removing rows with values greater than date
date = 5
df = df[df.VAr1 <date]


# In[29]:


df.sort_values(['VAr1'])


# In[46]:


data =df
data['shift']=  data['VAr1']- data['VAr1'].shift(periods=1)
data['shift/']=  data['VAr1']/ data['VAr1'].shift(periods=1)
data['shiftg']=  (data['VAr1']- data['VAr1'].shift(periods=1))-1
data['shiftg/']=  (data['VAr1']/ data['VAr1'].shift(periods=1))-1

data['shift4']= data['VAr1']- data['VAr1'].shift(4)
data['shift4/']= data['VAr1']/ data['VAr1'].shift(4)
data['shift4g']= (data['VAr1']- data['VAr1'].shift(4))-1
data['shift4g/']= (data['VAr1']/ data['VAr1'].shift(4))-1


# In[47]:


data.head()


# In[94]:


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


# In[96]:


df2= pd.read_csv('dummy2.csv')
df2.head()


# In[97]:


x =transformations(df2,'Vara')
x.head()


# In[90]:


x.columns


# In[100]:


x.columns[x.columns.str.contains(pat = 'logyd')]


# In[124]:


for c in x.columns:
    #print(c)
        #if (data[c].dtype != object) & ( ("qd" not in c ) ):
        if (data[c].dtype != object) & ( ("qd" not in c ) & ("qq" not in c )):
            print(c)


# In[126]:


import numpy as np
import pandas as pd
np.warnings.filterwarnings('ignore')
    
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
            data[name_qq] = ((data[c] /data[c].shift()) -1)*100
            
            # Year/year difference: (current year-last year)
            name_yd = str(c)+'_yd'
            data[name_yd] = data[c]-(data[c].shift(periods=4))
           
            # Year/year growth: (current year-last year)-1
            name_yy = str(c)+'_yy'
            data[name_yy] = ((data[c]/data[c].shift(periods=4)) -1 )*100
            
            
            # log of QQ difference
            name_logqd = str(c)+'_logqd'
            data[name_logqd] = np.log( data[c]) - np.log(data[c].shift() )
            
            
            ## try:
            name_logqd = str(c)+'_logqd2'
            data[name_logqd] = np.log( data[c] - data[c].shift() )
            
            
            
            # log of Quarter/Quarter growth: (current quarter-last quarter)-1
            name_logqq = str(c)+'_logqq'
            data[name_logqq] = ( np.log(data[c])  / np.log(data[c].shift()))-1 
            
            # log of Year/year difference: (current year-last year)-1
            name_logyd = str(c)+'_logyd'
            data[name_logyd] = (np.log(data[c]) - np.log(data[c].shift(periods=4)) )
           
            # log of Year/year growth: (current year-last year)-1
            name_logyy = str(c)+'_logyy'
            data[name_logyy] = (np.log( (data[c]))  / np.log(data[c].shift(periods=4))) -1  
            # negative values inside the log, which gives nan with real numbers
            
            
            # Lag1: Quarter/Quarter difference: current quarter-last quarter
            name_lagqd = str(c)+'_lagqd'
            data[name_lagqd] = data[c].shift(1)- data[c].shift(2)
            
            # Lag1: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lagqq = str(c)+'_lagqq'
            data[name_lagqq] = ((data[c].shift(1) /data[c].shift(2)) -1)*100
            
            # Lag1: Year/year difference: (current year-last year)
            name_lagyd = str(c)+'_lagyd'
            data[name_lagyd] = data[c].shift(4)-(data[c].shift(periods=8))
           
            # Lag1: Year/year growth: (current year-last year)-1
            name_lagyy = str(c)+'_lagyy'
            data[name_lagyy] = ((data[c].shift(4)/data[c].shift(periods=8)) -1 )*100
 

            # Lag2: Quarter/Quarter difference: current quarter-last quarter
            name_lag2qd = str(c)+'_lag2qd'
            data[name_lag2qd] = data[c].shift(2)- data[c].shift(3)
            
            # Lag2: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag2qq = str(c)+'_lag2qq'
            data[name_lag2qq] = ((data[c].shift(2) /data[c].shift(3)) -1)*100
            
            # Lag2: Year/year difference: (current year-last year)
            name_lag2yd = str(c)+'_lag2yd'
            data[name_lag2yd] = data[c].shift(8)-(data[c].shift(periods=12))
           
            # Lag2: Year/year growth: (current year-last year)-1
            name_lag2yy = str(c)+'_lag2yy'
            data[name_lag2yy] = ((data[c].shift(8)/data[c].shift(periods=12)) -1 )*100
 

            # Lag3: Quarter/Quarter difference: current quarter-last quarter
            name_lag3qd = str(c)+'_lag3qd'
            data[name_lag3qd] = data[c].shift(3)- data[c].shift(4)
            
            # Lag3: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag3qq = str(c)+'_lag3qq'
            data[name_lag3qq] = ((data[c].shift(3) /data[c].shift(4)) -1)*100
            
            # Lag3: Year/year difference: (current year-last year)
            name_lag3yd = str(c)+'_lag3yd'
            data[name_lag3yd] = data[c].shift(12)-(data[c].shift(periods=16))
           
            # Lag3: Year/year growth: (current year-last year)-1
            name_lag3yy = str(c)+'_lag3yy'
            data[name_lag3yy] = ((data[c].shift(12)/data[c].shift(periods=16)) -1 )*100
 

            # Lag4: Quarter/Quarter difference: current quarter-last quarter
            name_lag4qd = str(c)+'_lag4qd'
            data[name_lag4qd] = data[c].shift(4)- data[c].shift(5)
            
            # Lag4: Quarter/Quarter growth: (current quarter-last quarter)-1
            name_lag4qq = str(c)+'_lag4qq'
            data[name_lag4qq] = ((data[c].shift(4) /data[c].shift(5)) -1)*100
            
            # Lag4: Year/year difference: (current year-last year)
            name_lag4yd = str(c)+'_lag4yd'
            data[name_lag4yd] = data[c].shift(16)-(data[c].shift(periods=20))
           
            # Lag4: Year/year growth: (current year-last year)-1
            name_lag4yy = str(c)+'_lag4yy'
            data[name_lag4yy] = ((data[c].shift(16)/data[c].shift(periods=20)) -1 )*100    
    

    return data


# In[127]:


df2= pd.read_csv('dummy2.csv')
df2.head()


# In[128]:


df_transf =transformations(df2,'Vara')
df_transf.head(20)


# In[27]:


def lag_dif_noName(col,n=0,y=False):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    colname: string column name
    '''
    if n == 0:
        if y ==False:
            return col - col.shift(n+1)
        else: return col - col.shift(n+4)
    else:
        if y ==False:return col.shift(n) -col.shift(n+1)
        else: return col.shift(n) - col.shift(n+4)


# In[ ]:


def lag_growth_noName(col,n=0,y=False):
    ''' function that calculates the lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    '''
    if n == 0:
        if y ==False: return ((col /col.shift(n+1))-1)*100        
        else: return ((col /col.shift(n+4))-1)*100
        
    else:
        return  ((col.shift(n) /col.shift(n+1))-1)*100


# In[191]:


import numpy as np

def lag_difference(col,n=0,y=False,log=False, lag=False, colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    log= True for calculations of the log of the difference 
    lag= True for calculations of the lag of the difference
    ''' 
    # for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD' 
            
    elif y ==False and log==True and lag==False:
        value= np.log(col.shift(n)) -np.log(col.shift(n+1))
        colname = colname +'_logQD' 
    
    elif y ==False and log==False and lag==True:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_QD' +'Lag'+str(n)       
    
    # for Yearly calculations
    elif y ==True and log==False and lag==False:     
        value= col.shift(n) - col.shift(n+4)
        colname = colname +'_YD'  
    
    elif y ==True and log==True and lag==False:  
        value= np.log(col.shift(n)) - np.log(col.shift(n+4))
        colname = colname +'_logYD' 

    elif y ==True and log==False and lag==True:
        value= col.shift(n) -col.shift(n+1)
        colname = colname +'_YD' +'Lag'+str(n)          
        
        
    return colname, value


# In[205]:


def lag_growth(col, n=0, y=False, log=False,lag=False, colname='Column'):
    ''' function that calculates the lag n lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100 and returns an array
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    log= True for calculations of the log of the difference 
    ''' 
    # for Quarterly calculations
    if y ==False and log==False and lag==False:
        value= ((col.shift(n) /col.shift(n+1))-1)*100 
        colname = colname +'_QG'    
    
    elif y ==False and log==True and lag==False:
        value= (np.log(col.shift(n)) / np.log(col.shift(n+1)) )-1
        colname = colname +'_logQG'
        
    elif y ==False and log==False and lag==True:
        value= ((col.shift(n) /col.shift(n+1))-1)*100
        colname = colname +'_QG' +'Lag'+str(n)      
    
    # for Yearly calculations
    elif y ==True and log==False and lag==False:      
        value= ((col.shift(n) /col.shift(n+4))-1)*100
        colname = colname +'_YG'      
     
    elif y ==True and log==True and lag==False: 
        value= (np.log(col.shift(n)) /np.log(col.shift(n+4)) )-1
        colname = colname +'_YG' 
        
    elif y ==True and log==False and lag==True:
        value= ((col.shift(n) /col.shift(n+4))-1)*100
        colname = colname +'_YG' +'Lag'+str(n) 
        
    return colname, value


# In[206]:


def transformations(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):
            
            # Quarter/Quarter difference: current quarter-last quarter
            colname, data[colname] = lag_difference(data[c],colname= c)
            
            # Quarter/Quarter growth: (current quarter-last quarter)-1
            colname, data[colname] = lag_growth(data[c],colname= c)
                        
            # Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=False)
            
            # Year/year growth: (current year-last year)-1
            colname, data[colname] = lag_growth(data[c],colname= c, y=True)
            
            # log of Quarter/Quarter difference
            colname, data[colname] = lag_difference(data[c],colname= c, y=False, log=True)
            
            # log of Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=True)
            
            # LAGS:
            # Applying 1 to 4 lag transformation
            for i in range(1,n+1): 
                # Applying 1 to 4 lag transformation
                # for Quarter diff transformations:
                colname, data[colname] = lag_difference(data[c],colname= c, n=i,lag=True)
                
                # for yearly diff transformations
                colname, data[colname] = lag_difference(data[c],colname= c, n=i,lag=True, y=True)
                
                # for Quarter growth transformations:
                colname, data[colname] = lag_growth(data[c],colname= c, n=i,lag=True)
                
                # for yearly growth transformations
                colname, data[colname] = lag_growth(data[c],colname= c, n=i,lag=True, y=True)
            
    return data


# In[207]:


df2= pd.read_csv('dummy2.csv')
df2.head()


# In[208]:


df_transf =transformations(df2,'Vara')
df_transf.head(20)


# In[ ]:


#### PRACTICE
##################################################


# In[8]:


def lag_dif(col,n=0,q=0,colname='Column'):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    colname: string column name
    '''
    if n == 0:
        if q ==0:
            value = col - col.shift(n+1)
            colname2 = colname +'_QD'
             
        else: 
            value= col - col.shift(n+4)
            colname2 = colname +'_YD'
    else:
        if q ==0:
            value= col.shift(n) -col.shift(n+1)
            colname2 = colname +'_QD'
        else: 
            value= col.shift(n) - col.shift(n+4)
            colname2 = colname +'_YD'
            
    return value, colname2


# In[ ]:


def lag_growth(col,n=0,q=0):
    ''' function that calculates the lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    '''
    if n == 0:
        if q ==0: 
            value = ((col /col.shift(n+1))-1)*100  
            colname = colname +'_QG'
        else: 
            value= ((col /col.shift(n+4))-1)*100
            colname = colname +'_YG'
        
    else:
        if q ==0:
            value = ((col.shift(n) /col.shift(n+1))-1)*100
            colname = colname +'_QG'
        else:
            value= ((col.shift(n) /col.shift(n+4))-1)*100
            colname = colname +'_YG'


# In[ ]:


import numpy as np
np.warnings.filterwarnings('ignore')
    
def transformations(data,colSort,l=0):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort, l= number of lags to calculate
    '''
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):
            
            # Quarter/Quarter difference: current quarter-last quarter
            name, data[name] = lag_dif(data[c],n=0,q=0,colname= c)
            
            lag_dif(col,n=0,q=0,colname='Column'):
            
            #lags Quarter/Quarter difference:
            for i in range(1, n+1): 
            #lags Quarter/Quarter difference
                name_lagqd = str(c) +'_qd' +'_lag'+str(i)
                data[name_lagqd] = lag_dif(data[c],i)
                
                
            # Quarter/Quarter growth: (current quarter-last quarter)-1
            name_qq = str(c)+'_qq'
            data[name_qq] = lag_growth(data[c],0)
             
            #lags Quarter/Quarter growth:    
            for i in range(1, n+1): 
            #lags Quarter/Quarter difference
                name_lagqd = str(c) +'_qq' +'_lag'+str(i)
                data[name_lagqd] = lag_growth(data[c],i)               
                   
            
            # Year/year difference: (current year-last year)
            name_yd = str(c)+'_yd'
            data[name_yd] = data[c]-(data[c].shift(periods=4))
            
                for i in range(n+1): 
                #lags Year/year difference
                name_lagqd = str(c) +'_qd' +'_lag'+str(i+1)
                data[name_lagqd] = lag_n(data[c],i)
            
            
            
           
            # Year/year growth: (current year-last year)-1
            name_yy = str(c)+'_yy'
            data[name_yy] = ((data[c]/data[c].shift(periods=4)) -1 )*100
            
            
            # log of QQ difference
            name_logqd = str(c)+'_logqd'
            data[name_logqd] = np.log( data[c]) - np.log(data[c].shift() )
            
            # log of Quarter/Quarter growth: (current quarter-last quarter)-1
            name_logqq = str(c)+'_logqq'
            data[name_logqq] = ( np.log(data[c])  / np.log(data[c].shift())-1 )
            
            # log of Year/year difference: (current year-last year)-1
            name_logyd = str(c)+'_logyd'
            data[name_logyd] = (np.log(data[c]) - np.log(data[c].shift(periods=4)) )
           
            # log of Year/year growth: (current year-last year)-1
            name_logyy = str(c)+'_logyy'
            data[name_logyy] = (np.log( (data[c])  / np.log(data[c].shift(periods=4))) -1  )
            # negative values inside the log, which gives nan with real numbers
 

    return data


# In[ ]:


## working part


# In[52]:


def lag_dif_noName(col,n=0,y=False,log=False):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    
    '''
    if y ==False:return col.shift(n) -col.shift(n+1)
    else: return col.shift(n) - col.shift(n+4)  


# In[180]:


import numpy as np

def lag_difference(col,n=0,y=False,log=False,colname='Column',lag=False):
    ''' function that calculates the lag n difference: Q1 = X(t)  - X(t-1)
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    log= True for calculations of the log of the difference 
    ''' 
    # for Quarterly calculations
    if y ==False and log==False and lag==False:
        colname = colname +'_QD'  
        value= col.shift(n) -col.shift(n+1)
   
    if y ==False and log==False and lag==True:
        colname = colname +'_QD'+'Lag'+str(n)
        value= col.shift(n) -col.shift(n+1)
        
    else:
        pass
    return colname,value


# In[ ]:


def lag_growth_noName(col,n=0,y=False):
    ''' function that calculates the lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100
    col = array, n = interger,number of lags calculations. q:value==4 if is a YY calculation
    
    '''
    if y ==False: return ((col.shift(n) /col.shift(n+1))-1)*100        
    else: return ((col.shift(n) /col.shift(n+4))-1)*100


# In[81]:


def lag_growth(col,n=0,y=False,colname='Column'):
    ''' function that calculates the lag n lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100 and returns an array
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    log= True for calculations of the log of the difference 
    ''' 
    # for Quarterly calculations
    if y ==False:
            value= ((col.shift(n) /col.shift(n+1))-1)*100 
            colname = colname +'_QG'    
    
    # for Yearly calculations
    else:      
        value= ((col.shift(n) /col.shift(n+4))-1)*100
        colname = colname +'_YG'      
                 
    return colname, value


# In[ ]:





# In[130]:


def transformations(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object) & ( ("YY" not in c ) & ("QA" not in c )):
            
            # Quarter/Quarter difference: current quarter-last quarter
            colname, data[colname] = lag_difference(data[c],colname= c)
            
            # Quarter/Quarter growth: (current quarter-last quarter)-1
            colname, data[colname] = lag_growth(data[c],colname= c)
                        
            # Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=False)
            
            # Year/year growth: (current year-last year)-1
            colname, data[colname] = lag_growth(data[c],colname= c, y=True)
            
            # log of Quarter/Quarter difference
            colname, data[colname] = lag_difference(data[c],colname= c, y=False, log=True)
            
            # log of Year/year difference: (current year-last year)
            colname, data[colname] = lag_difference(data[c],colname= c, y=True, log=True)
            
            # LAGS:
            # Applying 1 to 4 lag transformation
            for i in range(1,n+1): 
                # for Quarter transformations:
                colname, data[colname+'_lag'+str(i)] = lag_difference(data[c],colname= c, n=i)
   
                # Quarter/Quarter growth: (current quarter-last quarter)-1
                colname, data[colname+'_lag'+str(i)] = lag_growth(data[c],colname= c, n=i)
            
            
    return data


# In[131]:


df2= pd.read_csv('dummy2.csv')
df2.head()


# In[132]:


df_transf =transformations(df2,'Vara')
df_transf.head(20)


# In[175]:


def lag_growth(col,n=0,y=False,colname='Column'):
    ''' function that calculates the lag n lag n growth:  Q1 = (( X(t) / X(t-1) )-1)*100 and returns an array
    col = array, n = interger,number of lags calculations. y:True(4Quarters) if it is a YY calculation
    colname: string column name
    log= True for calculations of the log of the difference 
    ''' 
    
    # for Quarterly calculations
    if y ==False:
            value= ((col.shift(n) /col.shift(n+1))-1)*100 
            colname = colname +'_QG'  
    else:
        pass
    return colname,value      


# In[176]:


def transformations2(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''    
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object):
            
            for i in range(1,n+1): 
                # Applying 1 to 4 lag transformation
                # for Quarter transformations:
                colname, data[colname+'_lag'+str(i)] = lag_difference(data[c],colname= c, n=i)
   
                # Quarter/Quarter growth: (current quarter-last quarter)-1
                colname, data[colname+'_lag'+str(i)] = lag_growth(data[c],colname= c, n=i)
    return data


# In[181]:


def transformations3(data,colSort,n=4):
    '''
    Function that sort and make lag transformations in variables returning a dataframe
    arg: data is a dataframe and colSort is a string value of the column to sort
    '''    
    data.sort_values(colSort,inplace=True) #sort values
    
    for c in data.columns:
        if (data[c].dtype != object):
            
            for i in range(1,n+1): 
                # Applying 1 to 4 lag transformation
                # for Quarter transformations:
                colname, data[colname] = lag_difference(data[c],colname= c, n=i,lag=True)
   
                # Quarter/Quarter growth: (current quarter-last quarter)-1
                #colname, data[colname+'_lag'+str(i)] = lag_growth(data[c],colname= c, n=i)
    return data


# In[182]:


df2= pd.read_csv('dummy2.csv')
df2.head()


# In[183]:


df_transf =transformations3(df2,'Vara')
df_transf.head(20)

