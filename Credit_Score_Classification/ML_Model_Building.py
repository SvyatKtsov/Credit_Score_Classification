
# importing libs for ML model building
# from sklearn.decomposition import PCA 
# pca = PCA(n_components=3)

# we can do this with RandomizedSeachCV to find an interval in which hyperparameters' values will be, 
                                                #   and then GridSearchCV to find the best hyperparameters from this interval

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score    # metrics for classification
from sklearn.model_selection import train_test_split as tts
import pickle as pckl
import time as t
# lib to save models
import joblib 


# # evaluating model using accuracy (for a classification task)
# print(f"============= ACCURACY:====================================: {accuracy_score(y_test, xgboost_classifier.predict(X_test))}  ") # y_true, y_pred
# print(f": {recall_score(y_test, xgboost_classifier.predict(X_test), average='micro')}")
# print(f": {f1_score(y_test, xgboost_classifier.predict(X_test), average='micro')}") # f1_score is also known as balanced F-score or F-measure

# et = t.time()
# print(f"=========TIME TAKEN TO FIT IPCA, MODEL AND EVALUATE EVERYTHING (50 000 rows):{et-st}")


with open('train_set_X.pkl', 'rb') as file:
    X = pckl.load(file)

with open('y_last_col.pkl', 'rb') as file:
    y = pckl.load(file)


# splitting data into train and test sets (using 'train_test_split')
X_train, X_test, y_train, y_test = tts(X,y, test_size=0.24, shuffle=True, random_state=79) # we need to get the transformed (one hot encoding) column 'Credit_Score'

random_forest_classifier = RandomForestClassifier(n_estimators=300,max_depth=20)
random_forest_classifier.fit(X_train, y_train)

#prediction:
print(f"Prediction for {X_test[0]}: {random_forest_classifier.predict(X_test[0].reshape(1,-1))}")





if __name__ == '__main__':
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

    from xgboost import XGBClassifier
    xgboost_classifier = XGBClassifier()

    
    
    xgboost_classifier.fit(X_train, y_train)

    # trying to fit different hypermarameters' values using RandomizedSearch and GridSearch
    random_forest_classifier = RandomForestClassifier()
    params_to_try = {'booster': ['gbtree', 'dart'], 'device':['cpu', 'gpu'],
                    'max_depth': [*range(10, 40, 10)], 'max_delta_step':[2,3], 
                    'lambda': [2,3, 0.4]}                                           ### for xgboost classifier
    
    params_to_try = {'n_estimators': [100, 300], 'max_depth':[40,60], 'min_samples_split': [4,5]}
    randomizedSearch_ = RandomizedSearchCV(random_forest_classifier, params_to_try, n_iter=4)
    randomizedSearch_.fit(X_train, y_train)


    
    
    #print(random_best_pars.best_estimator_, '\n' f'best parameters: {random_best_pars.best_params_}')
    print('\n' f'best parameters: {randomizedSearch_.best_params_}')
    #### n_iter - number of randomly selected combinations



    #time needed for everything: 
    st = t.time()

    xgboost_classifier  = XGBClassifier(max_depth=50, max_delta_step=2, device='cpu', booster='gbtree', reg_lambda=0.4)
    xgboost_classifier.fit(X_train, y_train, )
    print('\n Accuracy for XGBClassifier: ', accuracy_score(y_test, xgboost_classifier.predict(X_test)))
    print(f"Feature importances: {xgboost_classifier.feature_importances_}")

    
    grad_boost_classifier = GradientBoostingClassifier(max_depth=70, random_state=45)
    grad_boost_classifier.fit(X_train, y_train)
    print('\n Accuracy for GradBoostingClassfier: ', accuracy_score(y_test, grad_boost_classifier.predict(X_test)))
    print(f"Feature importances: {grad_boost_classifier.feature_importances_}");  print(f"GradBoosting score: {grad_boost_classifier.score}")



    dec_tree_classifier = DecisionTreeClassifier(max_depth=70, random_state=45)
    dec_tree_classifier.fit(X_train, y_train)
    print('\n Accuracy for DecTreeClassifier: ', accuracy_score(y_test, dec_tree_classifier.predict(X_test)))
    print(f"Feature importances: {dec_tree_classifier.feature_importances_}");  print(f"DecTree score: {dec_tree_classifier.score}")

    

    #random_forest_classifier = RandomForestClassifier(max_depth=70, random_state=45)
    random_forest_classifier = RandomForestClassifier(**randomizedSearch_.best_params_)
    random_forest_classifier.fit(X_train, y_train)
    print('\n Accuracy for RandomForestClassifier: ', accuracy_score(y_test, random_forest_classifier.predict(X_test)))
    print(f"Feature importances: {random_forest_classifier.feature_importances_}")



    ## saving the trained_model to a file .sav
    model_filename = 'Credit_Score_Classification_model.sav'
    joblib.dump(random_forest_classifier, model_filename)  ## saving the RandomForest model

    adaboost_classifier = AdaBoostClassifier(n_estimators=70, algorithm="SAMME", random_state=45)
    adaboost_classifier.fit(X_train, y_train)
    print('\n Accuracy for AdaBoostClassifier: ', accuracy_score(y_test, adaboost_classifier.predict(X_test))); print(f"AdaBoost score: {adaboost_classifier.score}")
    print(f"Feature importances: {adaboost_classifier.feature_importances_}")




    ########## your hardware may not be able to process stacking 
    # ensemble of an ensemble
    # stacking
    from sklearn.ensemble import StackingClassifier
    based_estimators = [('grad_boost', grad_boost_classifier), ('ada_boost', adaboost_classifier), ('xgboost', xgboost_classifier)]    
    stacking_classifier = StackingClassifier(estimators = based_estimators, final_estimator= XGBClassifier(max_depth=34, random_state=78))
    stacking_classifier.fit(X_train, y_train)
    print('\n Accuracy for StackingClassifier: ', accuracy_score(y_test, stacking_classifier.predict(X_test)))
    et = t.time()

    print(f"================================== TIME NEEDED FOR ALL THE .FIT AND TEST OPERATIONS: {(et-st)/60} minutes.......")
    print(f"Feature importances: {stacking_classifier.feature_importances_}")








