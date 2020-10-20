# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:38:53 2020

@author: Admin
"""

def process_sets(x):
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    # Retrieve numeric and categorical columns
    num_features = x._get_numeric_data().columns
    cat_features = list(set(x.columns) - set(num_features))
    
    # Impute num and cat features
    ii = IterativeImputer()
    x[num_features] = ii.fit_transform(x[num_features])
    si = SimpleImputer(strategy='most_frequent')
    x[cat_features] = si.fit_transform(x[cat_features])
    
    # Encode cat features
    x = pd.get_dummies(x, columns=cat_features, drop_first=True)
    
    # Scale X
    sc = StandardScaler()
    X_scaled = sc.fit_transform(x)
    x = pd.DataFrame(X_scaled, index=x.index, columns=x.columns)
    return x

def xgb_classifier(X_train, y_train, X_test):
    # Import XGBGeressor and GridSearchCV libraries
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV, RepeatedKFold

    parameters = {'max_depth': range (2, 10, 1),
                'n_estimators': range(60, 220, 40),
                'learning_rate': [0.1, 0.01, 0.05]
                 }

    grid = GridSearchCV(XGBClassifier(),
                            parameters,
                            cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1),
                            n_jobs = -1,
                            verbose=True)

    # Try fitting training data sets with all parameters
    grid.fit(X_train, y_train)

    # Print the best parameters
    print(grid.best_params_)
    
    # Fit the training tests using the best parameters
    best_grid = XGBClassifier(**grid.best_params_)
    best_grid.fit(X_train,y_train)

    # Predict y
    predictions = best_grid.predict(X_test)
    return predictions

def rfr_classifier(X_train, y_train, X_test):
    '''Advantage: It is invariant to scaled data'''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, RepeatedKFold

    # Create the parameter grid based on the results of random search 
    parameters = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    
    # Create a based model
    rf = RandomForestClassifier()
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = parameters, 
                              cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1), 
                              n_jobs = -1, verbose = 2)
    
    # Try fitting training data sets with all parameters
    grid_search.fit(X_train,y_train)
    
    # Print the best parameters
    print(grid_search.best_params_)
    
    #Fit the training tests using the best parameters
    best_grid = RandomForestClassifier(**grid_search.best_params_)
    best_grid.fit(X_train,y_train)
    
    # Get the predicted y
    predictions = best_grid.predict(X_test)
    
    # Print the score
    print('Score: %0.2f' %best_grid.score(X_train, y_train))
    return predictions

def Logistic_classifier(X_train, y_train, X_test):
    from sklearn.model_selection import GridSearchCV, RepeatedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    
    parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                  'class_weight': ['dict', 'balanced'],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'multi_class': ['auto', 'ovr', 'multinomial'],
                 }
    
    model = LogisticRegression(random_state=1)
    model_grid = GridSearchCV(model,
                            parameters,
                            cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1),
                            n_jobs = -1,
                            verbose=True)
    
    # Try fitting training data sets with all parameters
    model_grid.fit(X_train,y_train)
    
    # Print the best parameters
    print(model_grid.best_params_)
    
    #Fit the training tests using the best parameters
    best_search = LogisticRegression(**model_grid.best_params_, random_state=1)
    best_search.fit(X_train,y_train)
    
    # Print the score
    print('Score: %0.2f' %best_search.score(X_train, y_train))
    
    # Return predictions
    predictions = best_search.predict(X_test)
    return predictions

def svm_classifier(X_train, y_train, X_test):
    '''Disadvantage: Poor performance with large size data sets'''
    from sklearn.model_selection import GridSearchCV, RepeatedKFold
    from sklearn.svm import SVC
    
    parameters = {'C': range(13),
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': range(3, 6),
                  'gamma': ['scale', 'auto'],
                  'class_weight': ['dict', 'balanced'],
                  'decision_function_shape': ['ovo', 'ovr']
                 }
    
    model = SVC(random_state=1)
    model_grid = GridSearchCV(model,
                            parameters,
                            cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1),
                            n_jobs = -1,
                            verbose=True)
    
    # Try fitting training data sets with all parameters
    model_grid.fit(X_train,y_train)
    
    # Print the best parameters
    print(model_grid.best_params_)
    
    #Fit the training tests using the best parameters
    best_search = SVC(**model_grid.best_params_, random_state=1)
    best_search.fit(X_train,y_train)
    
    # Print the score
    print('Score: %0.2f' %best_search.score(X_train, y_train))
    
    # Return predictions
    predictions = best_search.predict(X_test)
    return predictions

def knn_classifier(X_train, y_train, X_test):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    
    parameters = {'n_neighbors':[5,6,7,8,9,10],
                  'leaf_size':[1,2,3,5],
                  'weights':['uniform', 'distance'],
                  'algorithm':['auto', 'ball_tree','kd_tree','brute']
                 }
    
    model = KNeighborsClassifier()
    model_grid = GridSearchCV(model,
                            parameters,
                            cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1),
                            n_jobs = -1,
                            verbose=True)
    
    # Try fitting training data sets with all parameters
    model_grid.fit(X_train,y_train)
    
    # Print the best parameters
    print(model_grid.best_params_)
    
    #Fit the training tests using the best parameters
    best_search = KNeighborsClassifier(**model_grid.best_params_)
    best_search.fit(X_train,y_train)
    
    # Print the score
    print('Score: %0.2f' %best_search.score(X_train, y_train))
    
    # Return predictions
    predictions = best_search.predict(X_test)
    return predictions

def xgb_regressor(X_train, y_train, X_test):
    # Import XGBGeressor and GridSearchCV libraries
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    parameters = {
                'max_depth': range (2, 10, 1),
                'n_estimators': range(60, 220, 40),
                'learning_rate': [0.1, 0.01, 0.05]
                 }

    grid = GridSearchCV(XGBRegressor(),
                            parameters,
                            cv = 5,
                            n_jobs = -1,
                            verbose=True)

    # Try fitting training data sets with all parameters
    grid.fit(X_train, y_train)

    # Print the best parameters
    print(grid.best_params_)
    
    # Fit the training tests using the best parameters
    best_grid = XGBRegressor(**grid.best_params_)
    best_grid.fit(X_train,y_train)

    # Predict y
    predictions = best_grid.predict(X_test)
    
    # Print the score
    print('Score: %0.2f' %best_grid.score(X_train, y_train))
    return predictions

def rfr_regressor(X_train, y_train, X_test):
    '''Advantage: It is invariant to scaled data'''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold

    # Create the parameter grid based on the results of random search 
    parameters = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    
    # Create a based model
    rf = RandomForestRegressor()
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = parameters, 
                              cv = KFold(5), n_jobs = -1, verbose = 2)
    
    # Try fitting training data sets with all parameters
    grid_search.fit(X_train,y_train)
    
    # Print the best parameters
    print(grid_search.best_params_)
    
    #Fit the training tests using the best parameters
    best_grid = RandomForestRegressor(**grid_search.best_params_)
    best_grid.fit(X_train,y_train)
    
    # Get the predicted y
    predictions = best_grid.predict(X_test)
    
    # Print the score
    print('Score: %0.2f' %best_grid.score(X_train, y_train))
    return predictions

def svm_regressor(X_train, y_train, X_test):
    '''Disadvantage: Poor performance with large size data sets'''
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold
    
    parameters = {'C': range(1, 13),
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'degree': range(3, 9),
                  'gamma': ['scale', 'auto'],
                  'coef0' : [0.01,10,0.5]
                 }
    
    model = SVR()
    model_grid = GridSearchCV(model,
                            parameters,
                            cv = KFold(5),
                            n_jobs = -1,
                            verbose=True)
    
    # Try fitting training data sets with all parameters
    model_grid.fit(X_train,y_train)
    
    # Print the best parameters
    print(model_grid.best_params_)
    
    #Fit the training tests using the best parameters
    best_search = SVR(**model_grid.best_params_, random_state=1)
    best_search.fit(X_train,y_train)
    
    # Print the score
    print('Score: %0.2f' %best_search.score(X_train, y_train))
    
    # Return predictions
    predictions = best_search.predict(X_test)
    return predictions

def ridge_regressor(X_train, y_train, X_test):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, KFold
    
    parameters = {'alpha': range(0, 101, 10),
                    'fit_intercept': [True,False], 
                    'normalize' :[False, True],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                 }
    
    model = Ridge()
    model_grid = GridSearchCV(model,
                            parameters,
                            cv = KFold(5),
                            n_jobs = -1,
                            verbose=True)
    
    # Try fitting training data sets with all parameters
    model_grid.fit(X_train,y_train)
    
    # Print the best parameters
    print(model_grid.best_params_)
    
    #Fit the training tests using the best parameters
    best_search = Ridge(**model_grid.best_params_)
    best_search.fit(X_train,y_train)
    
    # Print the score
    print('Score: %0.2f' %best_search.score(X_train, y_train))
    
    # Return predictions
    predictions = best_search.predict(X_test)
    return predictions

def classification_report(X_test, y_test, predictions, best_grid, display):
    from sklearn.metrics import classification_report, plot_confusion_matrix

    # Plot the confusion matrix
    plot_confusion_matrix(best_grid, X_test, y_test, display_labels=display)

    # return classification report 
    return(classification_report(y_test, predictions)) 

def test_imputations(X_train, y_train, num_cols):
    import statsmodels.api as sm
    from sklearn.impute import SimpleImputer
    from fancyimpute import KNN, IterativeImputer
    
    # Create an object of mean_imputed
    mean_imputer = SimpleImputer(strategy='mean')
    X_mean_imputed = sm.add_constant(mean_imputer.fit_transform(X_train[num_cols]))
    lm_mean = sm.OLS(y_train, X_mean_imputed).fit()

    # Create an object of median imputed
    median_imputer = SimpleImputer(strategy='median')
    X_median_imputed = sm.add_constant(median_imputer.fit_transform(X_train[num_cols]))
    lm_median = sm.OLS(y_train, X_median_imputed).fit()

    # Create an object of Iterative Imputer
    iterative_imputer = IterativeImputer()
    X_iterative_imputed = sm.add_constant(iterative_imputer.fit_transform(X_train[num_cols]))
    lm_iterative = sm.OLS(y_train, X_iterative_imputed).fit()

    # Create an object of KNN Imputer
    knn_imputer = KNN()
    X_knn_imputed = sm.add_constant(knn_imputer.fit_transform(X_train[num_cols]))
    lm_knn = sm.OLS(y_train, X_knn_imputed).fit()
    
    imputation_list = [('Mean', lm_mean.rsquared_adj), ('Median', lm_median.rsquared_adj), 
                       ('Iterative', lm_iterative.rsquared_adj), ('KNN', lm_knn.rsquared_adj)]
    imputation_list = sorted(imputation_list, key=(lambda x: x[1]), reverse=True)
    print('The best imputation: ', imputation_list[0])
    # print(pd.DataFrame({'Mean': lm_mean.rsquared_adj, 'Median': lm_median.rsquared_adj, 'Iterative': lm_iterative.rsquared_adj, 'KNN': lm_knn.rsquared_adj}, index=['R_squared_adj']))
    
def test_KNN_imputation(X_train, y_train, num_cols, neighbors):
    import statsmodels.api as sm
    from fancyimpute import KNN
    scores = []
    for k in neighbors:
        knn_imputer = KNN(k)
        X_knn_imputed = sm.add_constant(knn_imputer.fit_transform(X_train[num_cols]))
        lm_knn = sm.OLS(y_train, X_knn_imputed).fit()
        scores.append([lm_knn.rsquared_adj, 'k == ' + str(k)])
    sorted_scores = sorted(scores, key = lambda x: x[0], reverse=True)
    print(sorted_scores[0])
    
def evaluate_regression(X, y, model):
    '''Credit: Jason Brownlee
    Source: https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/'''
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    return abs(scores)

def compare_linear_regression(X, y):
    '''Credit: Jason Brownlee
    Source: https://machinelearningmastery.com/robust-regression-for-machine-learning-in-python/'''
    from numpy import mean
    from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
    result = []
    linear_list = [(LinearRegression(), 'linear'), (HuberRegressor(), 'huber'), (RANSACRegressor(), 'ransac'), (TheilSenRegressor(), 'theilsen')]
    for model in linear_list:
        result.append([mean(evaluate_regression(X, y, model[0])), model[1]])
    sorted_result = sorted(result, key=(lambda x: x[0]), reverse=True)
    print(sorted_result)

def compare_regression(X, y):
    from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor, Lasso, Ridge
    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold, cross_val_score
    models = [LinearRegression(), HuberRegressor(), RANSACRegressor(), TheilSenRegressor(), Lasso(), Ridge()]
    models.append(XGBRegressor())
    models.append(RandomForestRegressor())
    models.append(SVR())
    names = ['linear', 'huber', 'ransac', 'theilsen', 'lasso', 'ridge', 'xgb', 'rfr', 'svr']
    scoring = 'neg_mean_squared_error'
    results = []
    for model, name in zip(models, names):
    	kfold = KFold(n_splits=10)
    	cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
