#!/usr/bin/env python
# coding: utf-8

# ## Original Dataset (from Kaggle)
# 
# 
# https://www.kaggle.com/datasets/parisrohan/credit-score-classification

# In[1]:



# 1. EDA (Exploratory Data Analysis)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random as r


# In[2]:


#dtype = {'Customer_ID': str, 'Month': str, 'Name': str, 'Age': int, 'SSN':int}

train = pd.read_csv(r'D:\Kaggle_beginner_datasets_практика_\credit_score_\train.csv')
test = pd.read_csv(r'D:\Kaggle_beginner_datasets_практика_\credit_score_\test.csv') # train: D:\Kaggle_beginner_datasets_практика_\credit_score_\train.csv
print(pd.DataFrame(train)) 


# In[3]:


# cols to drop (train):
train.columns

#!!!    train_copy = train.drop(columns=['ID','Customer_ID','Name','SSN'])


# In[4]:


train['Delay_from_due_date'].describe()
sns.histplot(data=train['Delay_from_due_date']) # how many days the payment got delayed?

# there're no NaN values


# In[5]:


sns.histplot(data=train['Credit_Mix'])

# there're no NaN values (but values '-' can be considered NaN, so they would be deleted)


# In[6]:


train.Credit_Utilization_Ratio.describe()


# In[7]:


sns.histplot(data=train.Credit_Utilization_Ratio)

#credit utilization ratio, generally expressed as a percentage,
            #represents the amount of revolving credit you're using divided by the total credit available to you


# In[8]:


# h = ['Credit_History_Age',
#        'Payment_of_Min_Amount', 'Total_EMI_per_month',
#        'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
#        'Credit_Score']

# for each_last_feature in h:
#     sns.histplot(data=train[each_last_feature])


# In[9]:


h = ('Credit_History_Age',
       'Payment_of_Min_Amount', 'Total_EMI_per_month',
       'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
       'Credit_Score')

sns.histplot(data='Credit_History_Age')


# In[10]:


def get_categorical_features(dataset: pd.DataFrame) -> list:
    categorical_features_names_npArray = np.asarray(dataset.columns) # it will be faster with np.arrays()
    categorical_features_names = [each_name for each_name in categorical_features_names_npArray if type(dataset[each_name][0]) not in 
                                                              (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,) 
                                  and  type(dataset[each_name][30]) not in  (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,)]
    #print(len(categorical_features_names))
    return categorical_features_names, len(categorical_features_names) #num_of_categorical_features


# In[11]:


categ_feat_names, len_cfn = get_categorical_features(train)
len_cfn
non_categ_features = [feature for feature in list(train.columns) if feature not in categ_feat_names]
non_categ_features


# In[12]:


train.columns


# In[13]:


sns.histplot(data=train.Payment_Behaviour)


# In[14]:


train.Payment_Behaviour


# In[15]:


# Deleting NaN values from non-categorical columns

# for non-categorical columns, we can replace NaN values with the mean (.mean())
# and for categorical ones, we can perform random replacement (random.choice())


# ## Deleting NaN values from non-categorical columns
# 
# for non-categorical columns, we can replace NaN values with the mean

# In[16]:


non_categ_features #non-categorical attributes
# non_categ_features  = [float(o) for o in non_categ_features]

# other non-categ features which need to be transformed (from str to int/float):
num_features_to_transform = ('Age', 'Annual_Income', 'Num_of_Loan', 'Delay_from_due_date', 'Changed_Credit_Limit',
                             'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance', )


# In[17]:


categ_feat_names #categorical attributes


# In[18]:


num_features_to_transform


# In[19]:


train.Age.describe()
# 'top' is the most common value; 'freq' is the most common value's frequency.


# In[20]:


# for k, obj in enumerate(num_features_to_transform): # for each (feature to be transformed to float/int)
#     for obj in train[num_features_to_transform[k]]: # for each element in pd.Series (column)
#         if type(obj) != str and not obj.isdigit():
#             continue
#         else:
#             obj = float(obj)


# In[21]:


def is_valid_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

    
for feature in num_features_to_transform:  # for each feature to be transformed to float/int
    for i, obj in enumerate(train[feature]):  # for each element in pd.Series (column)
        if isinstance(obj, str):
            obj = obj.strip('_')  
        if is_valid_number(obj):
            train.at[i, feature] = float(obj)  
        else:
            print(f"Skipping invalid value: {obj} in column: {feature}")




# In[22]:


names, l = get_categorical_features(train)
names


# In[23]:


#Isinstance() is faster than type(), and it also considers inheritance. 
#In other words, it recognizes that an instance of a derived class is an instance of the base class as well. 
#That is why we often prefer isinstance() over type().



# Pandas methods to consider: 
    # pd.to_numeric()
    # pd.to_datetime()


# In[24]:


train.Credit_History_Age

# task: to write a function to convert this type: '31 Years and 6 Months' to this: 31.5 (this is years (31 +  6/12)) 


# In[25]:


def years_and_months_to_years(x: str) -> float:
    if pd.isna(x):
        return np.nan
    res = x.lower().split()
    years_ = int(res[0]) + int(res[3])/12
    return years_


# In[26]:


features_to_drop = (names[0], names[1], names[3], names[4]) # only personal data like ID, Name et cetera
features_to_drop


# In[27]:


train_copy = train.drop(columns=list(features_to_drop), inplace=False)
train_copy # dataset with all the necessary attributes


# In[28]:


train # original dataset



train_copy.columns  # len=24


# In[30]:


# train_copy.Type_of_Loan
# train_copy.Payment_Behaviour


train_copy.Type_of_Loan.value_counts()


# In[31]:


import re

def type_of_loan_to_norm(s: str) -> str:
    res = s.strip().replace('Loan', '')
    res = res.replace('and', '')
    res = res.replace(',', ''); #res = res.replace(' ', '')
    res= res.rstrip(); res = res.lstrip()

    res = ' '.join(res.split())
    return res

# in: Auto Loan, Credit-Builder Loan, Personal Loan
# out: Auto, Credir-Builder, Personal


# In[32]:


train_copy.Type_of_Loan


# In[33]:


type_of_loan_to_norm(train_copy.Type_of_Loan[0])


# In[34]:


print(type_of_loan_to_norm('Auto Loan, and Student Loan'))
type_of_loan_to_norm('Credit-Builder Loan')


# In[35]:


## the last function for categorical features' transformation
train_copy.Payment_Behaviour.value_counts()


# In[36]:


def payment_behavior(inp: str) -> str:    
    res = inp.replace('_', ' ')
    res = res.lower().split()

    if len(res) >= 3:
        res = res[0] + ' ' + res[2]
    else:
        res = inp  
    
    return res
    


# In[37]:


payment_behavior(train_copy.Payment_Behaviour[0])


# In[38]:


payment_behavior('High_spent_Small_value_payments')


# In[39]:

# In[40]:


# mask = train_copy['Num_Credit_Card'] % 2 ==0
# train_copy[~mask]


# In[41]:


names, l = get_categorical_features(train_copy) 
names


# In[42]:


non_categ_features = [feature for feature in list(train_copy.columns) if feature not in names]
print(non_categ_features)
train_copy.Monthly_Inhand_Salary       


# In[43]:


train_copy[non_categ_features].isnull().sum()


# In[44]:


# deleting NaNs in numeric columns 
numerical_cols_with_nans = ('Monthly_Inhand_Salary','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                           'Amount_invested_monthly', 'Monthly_Balance',)


# In[45]:


# type(train_copy.Monthly_Inhand_Salary[0])


# In[46]:


# ma=[]

# for nan_col in numerical_cols_with_nans:
#     col_mean = train_copy[nan_col].mean(skipna=True)
#     ma.append(col_mean)


# In[47]:


# ma=[]
# for nan_col in numerical_cols_with_nans:
#     train_copy[nan_col] = pd.to_numeric(train_copy[nan_col], errors='coerce')

# # Calculate means of each column ignoring NaN values

# for nan_col in numerical_cols_with_nans:
#     col_mean = train_copy[nan_col].mean(skipna=True)
#     ma.append(col_mean)

# print(ma)


# In[48]:


nan_cols_mean_values = []

train_copy.Num_of_Delayed_Payment.describe()[1]

nan_cols_mean_values = [train_copy[nan_col].describe()[1] for nan_col in numerical_cols_with_nans]
nan_cols_mean_values


# In[49]:


values_to_replace_nan = {key: item for key, item in zip(numerical_cols_with_nans,nan_cols_mean_values )}

train_copy.fillna(value=values_to_replace_nan,inplace=True)
train_copy.isnull().sum()


# In[50]:


train_copy.Occupation.value_counts()


# In[51]:


train_copy.Type_of_Loan                                            .value_counts()


# In[52]:


train_copy.Credit_History_Age           .value_counts()                


# In[53]:


train_copy.Credit_Score.isnull().sum()


# In[54]:


# Payment_Behaviour
def delete_inapprop_values_paymentBeh(val):
    if val == '!@9#%8':
        return r.choice(r_items)
    return val

r_items = [
    'Low_spent_Small_value_payments',  
    'High_spent_Medium_value_payments',    
    'Low_spent_Medium_value_payments',     
    'High_spent_Large_value_payments',     
    'High_spent_Small_value_payments',     
    'Low_spent_Large_value_payments'
]


train_copy = train_copy.applymap(delete_inapprop_values_paymentBeh)
train_copy


# In[55]:


train_copy.Payment_Behaviour.value_counts()


# In[56]:


# Payment_Behaviour
def delete_inapprop_values_all_cols(val: str):
    if val == '!@9#%8':
        return r.choice(payment_beh_values)
    elif val == '_':
        return r.choice(['Standard', 'Good', 'Bad'])
    elif val == '_______':
        return r.choice(all_occupation_names)
    return val

payment_beh_values = [
    'Low_spent_Small_value_payments',  
    'High_spent_Medium_value_payments',    
    'Low_spent_Medium_value_payments',     
    'High_spent_Large_value_payments',     
    'High_spent_Small_value_payments',     
    'Low_spent_Large_value_payments'
]



all_occupation_names = ['Lawyer' ,        
'Architect'  ,     
'Engineer'   ,      
'Scientist'  ,      
'Mechanic'    ,    
'Accountant'   ,   
'Developer'    ,    
'Media_Manager' ,  
'Teacher'       ,   
'Entrepreneur'  ,  
'Doctor'       ,   
'Journalist'    ,   
'Manager'      ,   
'Musician' ,   
'Writer']


# In[57]:


loan_types = ['Student Loan', 'Not Specified', 'Credit-Builder Loan', 'Personal Loan',
                                              'Debt Consolidation Loan']
def delete_type_of_loan_nan(val):
    if pd.isna(val):
        return r.choice(loan_types)
    return val


# In[58]:


train_copy = train_copy.applymap(delete_inapprop_values_all_cols)
train_copy


# In[59]:


train_copy['Type_of_Loan'] = train_copy['Type_of_Loan'].apply(delete_type_of_loan_nan)
train_copy['Type_of_Loan'].isnull().sum()


# In[60]:

train_copy.Credit_History_Age


# In[61]:


## the last column - 'Credit_History_Age'
# plan: using the written function, convert each str in this column to float and them take the mean; 
                                                                    #after - replace NaN with the mean
    

for each_str_val in train_copy.Credit_History_Age:
    each_str_val = years_and_months_to_years(each_str_val)
    
train_copy.Credit_History_Age


# In[62]:


type(train_copy.Credit_History_Age[0])


# In[63]:


train_copy['Credit_History_Age'] = train_copy['Credit_History_Age'].apply(years_and_months_to_years)


# In[64]:


sns.histplot(data=train_copy.Credit_History_Age)


# In[65]:


mask = train_copy['Credit_History_Age'].isna()
train_copy[~mask]


# In[66]:


# # deleting NaN using mean of (train_copy['Credit_History_Age']):

# train_copy['Credit_History_Age'].fillna(train_copy['Credit_History_Age'].describe()[1])

# train_copy.isnull().sum()


# In[67]:


def years_and_months_to_years(x):
    if pd.isna(x):
        return np.nan
    elif isinstance(x, str): 
        res = x.lower().split()
        years_ = int(res[0]) + int(res[3])/12
        return years_
    else:  
        return x


train_copy['Credit_History_Age'] = train_copy['Credit_History_Age'].apply(years_and_months_to_years)
mean_age = train_copy['Credit_History_Age'].mean()
train_copy['Credit_History_Age'].fillna(mean_age, inplace=True)
print(train_copy.isnull().sum())


# In[68]:


train_copy.isnull().sum()

## no NaN values + everything is working great
sns.histplot(data=train_copy.Credit_History_Age)


# In[69]:


mask = train_copy['Credit_History_Age'].isna()
train_copy[~mask]

# In[70]:


categ_features, ln = get_categorical_features(train_copy)
categ_features


# In[71]:


type(train_copy.values) # np.ndarray
train_copy.values


# In[72]:


non_categ_features = [feature for feature in list(train_copy.columns) if feature not in categ_features]
non_categ_features


# In[73]:


# plotting a correlation heatmap:

sns.heatmap(train_copy[non_categ_features].corr(), annot=False)


# In[74]:

# In[75]:


non_categ_features # no need to use StandardScaler (some outliers...)
# we can just fit it to a model already


# In[76]:


categ_features # we should use OneHotEncoder or LabelEncoder etc.


# ### OneHotEncoder for non-ordinal categorical attributes

# In[77]:


train_copy.Credit_Mix.value_counts()


# In[78]:


train_copy.Credit_Score.value_counts()


# In[79]:


new_train_copy = train_copy.copy()
#new_train_copy


# In[80]:


from sklearn import preprocessing as prepr
le = prepr.LabelEncoder()


# In[81]:
new_train_copy['Credit_Score'].value_counts()


# In[82]:


new_train_copy.Credit_Mix.value_counts()


# In[83]:

# In[84]:


new_train_copy.Credit_Mix.value_counts()


# ### One Hot Encoding (nominal (non-ordinal) features)
# 
# Month
# 
# Occupation
# 
# Type_of_Loan
# 
# Payment_of_Min_Amount
# 
# Payment_Behaviour

# In[85]:


from sklearn import preprocessing as prepr

onehotencoder = prepr.OneHotEncoder()

# reshape the 1-D country array to 2-D as fit_transform expects 2-D and fit the encoder
X = onehotencoder.fit_transform(new_train_copy.Month.values.reshape(-1, 1)).toarray()

all_months = ["January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August"]
 
# columns=["Month_" + str(int(i)) for i in range(X.shape[1])
new_train_copy_OneHot = pd.DataFrame(X, columns=all_months)
new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
new_train_copy = new_train_copy.drop(['Month'], axis=1)


# In[86]:


#new_train_copy.Month.value_counts()

# error because this col has been deleted


# In[87]:


new_train_copy.drop(columns=['August'],inplace=True)


# In[88]:


# ## Occupation

X = onehotencoder.fit_transform(new_train_copy.Occupation.values.reshape(-1, 1)).toarray()

all_occupations = ['Scientist', 'Teacher', 'Developer', 'Engineer', 'Journalist',
       'Mechanic', 'Entrepreneur', 'Lawyer', 'Manager', 'Media_Manager',
       'Doctor', 'Writer', 'Accountant', 'Musician', 'Architect']
 
new_train_copy_OneHot = pd.DataFrame(X, columns=all_occupations)
new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)

new_train_copy = new_train_copy.drop(['Occupation'], axis=1)

#new_train_copy.Occupation.unique()


# In[89]:


new_train_copy.drop(columns=['Architect'],inplace=True)


# In[90]:


new_train_copy


# In[91]:


categ_features


# In[92]:


print(len(new_train_copy.Type_of_Loan.unique()))
new_train_copy.Type_of_Loan.value_counts()


# In[93]:


new_train_copy.Payment_of_Min_Amount.value_counts()


# In[94]:


new_train_copy.Payment_Behaviour.value_counts()


# In[95]:


## Payment_of_Min_Amount

X = onehotencoder.fit_transform(new_train_copy.Payment_of_Min_Amount.values.reshape(-1, 1)).toarray()

all_payment_minAmount = ['Yes', 'No', 'NM']
 
new_train_copy_OneHot = pd.DataFrame(X, columns=all_payment_minAmount)

new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)

new_train_copy = new_train_copy.drop(['Payment_of_Min_Amount'], axis=1)


# In[96]:


new_train_copy.drop(columns=['NM'],inplace=True)


# In[107]:


new_train_copy


# In[98]:


type_of_loan_to_norm('Debt Consolidation Loan, and Auto Loan, Personal Loan, Debt Consolidation Loan, Student Loan, and Credit-Builder Loan')
#payment_behavior()


# In[99]:

payment_behavior('Low_spent_Large_value_payments')



lst_ = []

for h in list(new_train_copy.Type_of_Loan.unique()):
    lst_.append(type_of_loan_to_norm(h))


# ### doing One Hot Encoding on a feature with 6500+- unique values

# In[100]:


# ## Type_of_Loan

# X = onehotencoder.fit_transform(new_train_copy.Type_of_Loan.values.reshape(-1, 1)).toarray()

# all_loan_types = lst_
 
# new_train_copy_OneHot = pd.DataFrame(X, columns=all_loan_types)


# new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)

# new_train_copy = new_train_copy.drop(['Type_of_Loan'], axis=1)


# In[101]:


# new_train_copy.drop(columns=[lst_[-1]],inplace=True)


# In[102]:


new_train_copy


# In[108]:


# import gc 
# gc.collect()

payment_behavior('Low_spent_Large_value_payments')

#lst_ = list(new_train_copy.Type_of_Loan.unique())

lst_paym_beh = []

for k in list(new_train_copy.Payment_Behaviour.unique()):
    lst_paym_beh.append(payment_behavior(k))


# In[111]:


## Payment_behavior

X = onehotencoder.fit_transform(new_train_copy.Payment_Behaviour.values.reshape(-1, 1)).toarray()
all_payment_behavior = lst_paym_beh
new_train_copy_OneHot = pd.DataFrame(X, columns=all_payment_behavior)
new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
new_train_copy = new_train_copy.drop(['Payment_Behaviour'], axis=1)


# In[112]:


new_train_copy.drop(columns=[lst_paym_beh[-1]],inplace=True)


# In[113]:


new_train_copy


# ### dataset is ready (except for the col 'Type_of_Loan')

# In[ ]:




