import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random as r


# In[2]:



if __name__ == '__main__':
    train = pd.read_csv(r'D:\Kaggle_beginner_datasets_практика_\credit_score_\train.csv')
    train = train.iloc[:50000, :]
    test = pd.read_csv(r'D:\Kaggle_beginner_datasets_практика_\credit_score_\test.csv') 


    def get_categorical_features(dataset: pd.DataFrame) -> list:
        categorical_features_names_npArray = np.asarray(dataset.columns) 
        categorical_features_names = [each_name for each_name in categorical_features_names_npArray if type(dataset[each_name][0]) not in 
                                                                (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,) 
                                    and  type(dataset[each_name][30]) not in  (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,)]
        return categorical_features_names, len(categorical_features_names) 


    categ_feat_names, len_cfn = get_categorical_features(train)
    non_categ_features = [feature for feature in list(train.columns) if feature not in categ_feat_names]

    num_features_to_transform = ('Age', 'Annual_Income', 'Num_of_Loan', 'Delay_from_due_date', 'Changed_Credit_Limit',
                                'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance', )

    def is_valid_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

        
    for feature in num_features_to_transform:  
        for i, obj in enumerate(train[feature]):  
            if isinstance(obj, str):
                obj = obj.strip('_') 
            if is_valid_number(obj):
                train.at[i, feature] = float(obj) 
            else:
                print(f"Skipping invalid value: {obj} in column: {feature}")



    names, l = get_categorical_features(train)


    def years_and_months_to_years(x: str) -> float:
        if pd.isna(x):
            return np.nan
        res = x.lower().split()
        years_ = int(res[0]) + int(res[3])/12
        return years_


    # In[26]:


    features_to_drop = (names[0], names[1], names[3], names[4])

    train_copy = train.drop(columns=list(features_to_drop), inplace=False)


    import re
    def type_of_loan_to_norm(s: str) -> str:

        res = s.strip().replace('Loan', '')
        res = res.replace('and', '')
        res = res.replace(',', ''); #res = res.replace(' ', '')
        res= res.rstrip(); res = res.lstrip()

        res = ' '.join(res.split())
        return res


    def payment_behavior(inp: str) -> str:
        
        res = inp.replace('_', ' ')

        res = res.lower().split()

        if len(res) >= 3:
            res = res[0] + ' ' + res[2]
        else:
            res = inp 
        
        
        return res
        


    names, l = get_categorical_features(train_copy)
    non_categ_features = [feature for feature in list(train_copy.columns) if feature not in names]

    numerical_cols_with_nans = ('Monthly_Inhand_Salary','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                            'Amount_invested_monthly', 'Monthly_Balance',)


    nan_cols_mean_values = [train_copy[nan_col].describe()[1] for nan_col in numerical_cols_with_nans]
    values_to_replace_nan = {key: item for key, item in zip(numerical_cols_with_nans,nan_cols_mean_values )}
    train_copy.fillna(value=values_to_replace_nan,inplace=True)

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

    print(type(train_copy))
    train_copy = train_copy.applymap(delete_inapprop_values_paymentBeh)

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
        'Low_spent_Small_value_payments',  'High_spent_Medium_value_payments',    'Low_spent_Medium_value_payments',     'High_spent_Large_value_payments',     'High_spent_Small_value_payments',     'Low_spent_Large_value_payments'
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



    loan_types = ['Student Loan', 'Not Specified', 'Credit-Builder Loan', 'Personal Loan',
                                                'Debt Consolidation Loan']
    def delete_type_of_loan_nan(val):
        if pd.isna(val):
            return r.choice(loan_types)
        return val



    train_copy = train_copy.applymap(delete_inapprop_values_all_cols)
    train_copy['Type_of_Loan'] = train_copy['Type_of_Loan'].apply(delete_type_of_loan_nan)


    for each_str_val in train_copy.Credit_History_Age:
        each_str_val = years_and_months_to_years(each_str_val)


    train_copy['Credit_History_Age'] = train_copy['Credit_History_Age'].apply(years_and_months_to_years)



    def years_and_months_to_years(x):
        if pd.isna(x):
            return np.nan
        elif isinstance(x, str):  
            res = x.lower().split()
            years_ = int(res[0]) + int(res[3])/12
            return years_
        else:  #
            return x


    train_copy['Credit_History_Age'] = train_copy['Credit_History_Age'].apply(years_and_months_to_years)
    mean_age = train_copy['Credit_History_Age'].mean()
    train_copy['Credit_History_Age'].fillna(mean_age, inplace=True)



    categ_features, ln = get_categorical_features(train_copy)
    categ_features


    non_categ_features = [feature for feature in list(train_copy.columns) if feature not in categ_features]
    non_categ_features
    new_train_copy = train_copy.copy()


    from sklearn import preprocessing as prepr
    le = prepr.LabelEncoder()
    new_train_copy['Credit_Score'] = new_train_copy['Credit_Score'].apply(lambda x: ["Poor", "Standard", "Good"].index(x))
    new_train_copy['Credit_Mix'] = new_train_copy['Credit_Mix'].apply(lambda x: ["Bad", "Standard", "Good"].index(x))



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


    from sklearn import preprocessing as prepr
    onehotencoder = prepr.OneHotEncoder()

    X = onehotencoder.fit_transform(new_train_copy.Month.values.reshape(-1, 1)).toarray()
    all_months = ["January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August"]


    new_train_copy_OneHot = pd.DataFrame(X, columns=all_months)
    new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
    new_train_copy.drop(columns=['Month'], inplace=True)

    new_train_copy.drop(columns=['August'],inplace=True)



    # ## Occupation
    X = onehotencoder.fit_transform(new_train_copy.Occupation.values.reshape(-1, 1)).toarray()
    all_occupations = ['Scientist', 'Teacher', 'Developer', 'Engineer', 'Journalist',
        'Mechanic', 'Entrepreneur', 'Lawyer', 'Manager', 'Media_Manager',
        'Doctor', 'Writer', 'Accountant', 'Musician', 'Architect']

    new_train_copy_OneHot = pd.DataFrame(X, columns=all_occupations)
    new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
    new_train_copy = new_train_copy.drop(['Occupation'], axis=1)

    new_train_copy.drop(columns=['Architect'],inplace=True)



    ## Payment_of_Min_Amount
    X = onehotencoder.fit_transform(new_train_copy.Payment_of_Min_Amount.values.reshape(-1, 1)).toarray()
    all_payment_minAmount = ['Yes', 'No', 'NM']
    new_train_copy_OneHot = pd.DataFrame(X, columns=all_payment_minAmount)
    new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
    new_train_copy = new_train_copy.drop(['Payment_of_Min_Amount'], axis=1)

    new_train_copy.drop(columns=['NM'],inplace=True)


    type_of_loan_to_norm('Debt Consolidation Loan, and Auto Loan, Personal Loan, Debt Consolidation Loan, Student Loan, and Credit-Builder Loan')
    payment_behavior('Low_spent_Large_value_payments')



    # import gc 
    # gc.collect()
    payment_behavior('Low_spent_Large_value_payments')


    lst_paym_beh = []
    for k in list(new_train_copy.Payment_Behaviour.unique()):
        lst_paym_beh.append(payment_behavior(k))

    ## Payment_behavior

    X = onehotencoder.fit_transform(new_train_copy.Payment_Behaviour.values.reshape(-1, 1)).toarray()
    all_payment_behavior = lst_paym_beh
    new_train_copy_OneHot = pd.DataFrame(X, columns=all_payment_behavior)
    new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
    new_train_copy = new_train_copy.drop(['Payment_Behaviour'], axis=1)
    new_train_copy.drop(columns=[lst_paym_beh[-1]],inplace=True)




    ## Type_of_Loan

    lst_ = []
    for h in list(new_train_copy.Type_of_Loan.unique()):
        lst_.append(type_of_loan_to_norm(h))
    # ### doing One Hot Encoding on a feature with 6500+- unique values

    X = onehotencoder.fit_transform(new_train_copy.Type_of_Loan.values.reshape(-1, 1)).toarray()
    new_train_copy_OneHot = pd.DataFrame(X, columns=lst_)
    new_train_copy = pd.concat([new_train_copy, new_train_copy_OneHot], axis=1)
    new_train_copy = new_train_copy.drop(['Type_of_Loan'], axis=1)

    new_train_copy.drop(columns=[lst_[-1]],inplace=True)

    print(new_train_copy) # the last column which needs One Hot Encoding
    print(new_train_copy.columns)
    new_train_copy_x = new_train_copy.iloc[:, :-1]







    # ######### ML Model Training

    import time as t

    st = t.time()
    from sklearn.decomposition import IncrementalPCA # lets you fir the model in mini-batches 


    X = new_train_copy.iloc[:, :-1]
    n_components = 9  

    n = 50000  
    chunk_size = 1000 


    numeric_data = new_train_copy_x.apply(pd.to_numeric, errors='coerce') 
    numeric_data = numeric_data.dropna(axis=1, how='any')  
    data = numeric_data.values 

    ipca = IncrementalPCA(n_components=n_components, batch_size=16)

    for i in range(0, n // chunk_size):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n)  
        ipca.partial_fit(data[start_idx:end_idx])
        
    print(f"ipca.explained_variance_: {ipca.explained_variance_}, ipca.components_:{ipca.components_}, ipca.singular_values_: {ipca.singular_values_}")
    print(f"\n\nipca.explained_variance_ratio_: {ipca.explained_variance_ratio_}")
    information_explained_in_percent = sum([each_PC_variance*100 for each_PC_variance in ipca.explained_variance_ratio_])
    print(f"How much, in percent, info IPCA described: {information_explained_in_percent}")


    data_transformed = ipca.transform(data)
    import pickle as pckl
    with open('train_set_X.pkl', 'wb') as file:
        pckl.dump(data_transformed, file)

    with open('y_last_col.pkl', 'wb') as file:
        pckl.dump(new_train_copy['Credit_Score'], file)


# __all__ = ['pd', 'np', 'plt', 'r', 'sns']

