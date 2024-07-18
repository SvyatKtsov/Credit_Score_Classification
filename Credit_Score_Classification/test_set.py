import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import random as r

#from train_set import test

if __name__ == '__main__':
    test = pd.read_csv(r'D:\Kaggle_beginner_datasets_практика_\credit_score_\test.csv')
    print(test.columns)



    def get_categorical_features(dataset: pd.DataFrame) -> list:
        categorical_features_names_npArray = np.asarray(dataset.columns) 
        categorical_features_names = [each_name for each_name in categorical_features_names_npArray if type(dataset[each_name][0]) not in 
                                                                (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,) 
                                    and  type(dataset[each_name][30]) not in  (int, float, np.ndarray, np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,)]

        return categorical_features_names, len(categorical_features_names)


    categ_feat_names, len_cfn = get_categorical_features(test)
    non_categ_features = [feature for feature in list(test.columns) if feature not in categ_feat_names]

    num_features_to_transform = ('Age', 'Annual_Income', 'Num_of_Loan', 'Delay_from_due_date', 'Changed_Credit_Limit',
                                'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance', )

    def is_valid_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

        
    for feature in num_features_to_transform:  
        for i, obj in enumerate(test[feature]):  
            if isinstance(obj, str):
                obj = obj.strip('_')  
            if is_valid_number(obj):
                test.at[i, feature] = float(obj) 
            else:
                print(f"Skipping invalid value: {obj} in column: {feature}")


    names, l = get_categorical_features(test)


    def years_and_months_to_years(x: str) -> float:
        if pd.isna(x):
            return np.nan
        res = x.lower().split()
        years_ = int(res[0]) + int(res[3])/12
        return years_


    # In[26]:


    features_to_drop = (names[0], names[1], names[3], names[4]) 
 

    test_copy = test.copy()
    test.drop(columns=list(features_to_drop), inplace=True)
    print(test_copy.columns) 


    import re
    def type_of_loan_to_norm(s: str) -> str:

        res = s.strip().replace('Loan', '')
        res = res.replace('and', '')
        res = res.replace(',', ''); 
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
        


    names, l = get_categorical_features(test_copy)
    non_categ_features = [feature for feature in list(test_copy.columns) if feature not in names]


    numerical_cols_with_nans = ('Monthly_Inhand_Salary','Num_of_Delayed_Payment','Num_Credit_Inquiries',
                            'Amount_invested_monthly', 'Monthly_Balance',)


    nan_cols_mean_values = [test_copy[nan_col].describe()[1] for nan_col in numerical_cols_with_nans]
    values_to_replace_nan = {key: item for key, item in zip(numerical_cols_with_nans,nan_cols_mean_values )}
    test_copy.fillna(value=values_to_replace_nan,inplace=True)
    print(test_copy.columns) 

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


    test_copy.applymap(delete_inapprop_values_paymentBeh)

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

    all_occupation_names = ['Scientist',
    'Teacher','Engineer','Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']


    loan_types = ['Student Loan', 'Not Specified', 'Credit-Builder Loan', 'Personal Loan',
                                                'Debt Consolidation Loan'] 
    def delete_type_of_loan_nan(val):
        if pd.isna(val):
            return r.choice(loan_types)
        return val




    print(test_copy.columns)
    test_copy['Type_of_Loan'] = test_copy['Type_of_Loan'].apply(delete_type_of_loan_nan)
    print(test_copy.columns)

    for each_str_val in test_copy.Credit_History_Age:
        each_str_val = years_and_months_to_years(each_str_val)


    test_copy['Credit_History_Age'] = test_copy['Credit_History_Age'].apply(years_and_months_to_years)
    print(test_copy.columns)


    def years_and_months_to_years(x):
        if pd.isna(x):
            return np.nan
        elif isinstance(x, str):  
            res = x.lower().split()
            years_ = int(res[0]) + int(res[3])/12
            return years_
        else:  
            return x



    test_copy['Credit_History_Age'] = test_copy['Credit_History_Age'].apply(years_and_months_to_years)
    mean_age = test_copy['Credit_History_Age'].mean()
    test_copy['Credit_History_Age'].fillna(mean_age, inplace=True)


    categ_features, ln = get_categorical_features(test_copy)
    categ_features

    non_categ_features = [feature for feature in list(test_copy.columns) if feature not in categ_features]
    non_categ_features
    new_test_copy = test_copy.copy()


    from sklearn import preprocessing as prepr
    le = prepr.LabelEncoder()
    print(new_test_copy['Credit_Mix'].value_counts(),'\n')


    # Credit_Mix
    def delete_inapprop_values_CreditMix(val): 
        if val == '_':
            return r.choice(cr_mix)
        return val

    cr_mix = ['Bad', 'Standard', 'Good']

    new_test_copy = new_test_copy.applymap(delete_inapprop_values_CreditMix)
    new_test_copy['Credit_Mix'] = new_test_copy['Credit_Mix'].apply(lambda x: ["Bad", "Standard", "Good"].index(x))



    from sklearn import preprocessing as prepr
    onehotencoder = prepr.OneHotEncoder()

    X = onehotencoder.fit_transform(new_test_copy.Month.values.reshape(-1, 1)).toarray()
    all_months = ['September', 'October', 'November', 'December']
    

    new_test_copy_OneHot = pd.DataFrame(X, columns=all_months)
    new_test_copy = pd.concat([new_test_copy, new_test_copy_OneHot], axis=1)
    new_test_copy.drop(columns=['Month'], inplace=True)
    new_test_copy.drop(columns=['December'],inplace=True)


    new_test_copy = new_test_copy.applymap(delete_inapprop_values_all_cols)

    print('+++++++++++++++++++++++++++++++++ ================= new_test_copy.columns: ',  new_test_copy.columns, '\n')
    # ## Occupation
    X = onehotencoder.fit_transform(new_test_copy.Occupation.values.reshape(-1, 1)).toarray()
    all_occupations = ['Scientist', 'Teacher', 'Developer', 'Engineer', 'Journalist',
        'Mechanic', 'Entrepreneur', 'Lawyer', 'Manager', 'Media_Manager',
        'Doctor', 'Writer', 'Accountant', 'Musician', 'Architect']
    
    new_test_copy_OneHot = pd.DataFrame(X, columns=all_occupations)
    new_test_copy = pd.concat([new_test_copy, new_test_copy_OneHot], axis=1)
    new_test_copy = new_test_copy.drop(['Occupation'], axis=1)
    new_test_copy.drop(columns=['Architect'],inplace=True)



    ## Payment_of_Min_Amount
    X = onehotencoder.fit_transform(new_test_copy.Payment_of_Min_Amount.values.reshape(-1, 1)).toarray()
    all_payment_minAmount = ['Yes', 'No', 'NM']
    new_test_copy_OneHot = pd.DataFrame(X, columns=all_payment_minAmount)
    new_test_copy = pd.concat([new_test_copy, new_test_copy_OneHot], axis=1)
    new_test_copy = new_test_copy.drop(['Payment_of_Min_Amount'], axis=1)
    new_test_copy.drop(columns=['NM'],inplace=True)


    lst_paym_beh = []
    for k in list(new_test_copy.Payment_Behaviour.unique()):
        lst_paym_beh.append(payment_behavior(k))

    ## Payment_behavior

    X = onehotencoder.fit_transform(new_test_copy.Payment_Behaviour.values.reshape(-1, 1)).toarray()
    all_payment_behavior = lst_paym_beh
    new_test_copy_OneHot = pd.DataFrame(X, columns=all_payment_behavior)
    new_test_copy = pd.concat([new_test_copy, new_test_copy_OneHot], axis=1)
    new_test_copy = new_test_copy.drop(['Payment_Behaviour'], axis=1)
    new_test_copy.drop(columns=[lst_paym_beh[-1]],inplace=True)




    ## Type_of_Loan
    lst_ = []
    for h in list(new_test_copy.Type_of_Loan.unique()):
        lst_.append(type_of_loan_to_norm(h))

    X = onehotencoder.fit_transform(new_test_copy.Type_of_Loan.values.reshape(-1, 1)).toarray()
    new_test_copy_OneHot = pd.DataFrame(X, columns=lst_)
    new_test_copy = pd.concat([new_test_copy, new_test_copy_OneHot], axis=1)
    new_test_copy = new_test_copy.drop(['Type_of_Loan'], axis=1)
    new_test_copy.drop(columns=[lst_[-1]],inplace=True)



    # deleteing personal-info features:
    #pers_info_feat = ['ID', 'Customer_ID', 'Name', 'SSN']
    new_test_copy.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN'], inplace=True)
    print(new_test_copy)
    print(f"======columns are: ======: {list(new_test_copy.columns)[:22]}")   ### done
    print('------------------------------------------------------------------------------------------')
    










