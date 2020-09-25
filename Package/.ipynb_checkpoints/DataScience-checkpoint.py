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
    from sklearn.model_selection import GridSearchCV

    parameters = {'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': range(2, 10),
                'scoring': ['roc_auc'], 
                'learning_rate': [0.1, 0.01, 0.05],
                'n_estimators': range(60, 220, 40),
                 }

    grid = GridSearchCV(XGBClassifier(),
                            parameters,
                            cv = 5,
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
    return (best_grid, predictions)

def xgb_regressor(X_train, y_train, X_test):
    # Import XGBGeressor and GridSearchCV libraries
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV

    parameters = {'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': range(2, 10),
                'scoring': ['roc_auc'], 
                'learning_rate': [0.1, 0.01, 0.05],
                'n_estimators': range(60, 220, 40),
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
    return (best_grid, predictions)

def classification_report(X_test, y_test, predictions, best_grid, display):
    from sklearn.metrics import classification_report, plot_confusion_matrix

    # Plot the confusion matrix
    plot_confusion_matrix(best_grid, X_test, y_test, display_labels=display)

    # return classification report 
    return(classification_report(y_test, predictions)) 

def rfr_classifier(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

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
                              cv = 5, n_jobs = -1, verbose = 2)
    
    # Try fitting training data sets with all parameters
    grid_search.fit(X_train,y_train)
    
    # Print the best parameters
    print(grid_search.best_params_)
    
    #Fit the training tests using the best parameters
    best_grid = RandomForestClassifier(**grid_search.best_params_)
    best_grid.fit(X_train,y_train)
    
    # Get the predicted y
    predictions = best_grid.predict(X_test)
    return (best_grid, predictions)

def test_imputations(X_train, y_train, num_cols):
    import statsmodels.api as sm
    import pandas as pd
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

    print(pd.DataFrame({'Mean': lm_mean.rsquared_adj, 'Median': lm_median.rsquared_adj, 'Iterative': lm_iterative.rsquared_adj, 'KNN': lm_knn.rsquared_adj}, index=['R_squared_adj']))
    
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
