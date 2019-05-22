'''
ML loop
'''


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy,\
    precision_recall_fscore_support, classification_report, roc_auc_score, \
    precision_recall_curve, confusion_matrix, recall_score
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
BaggingClassifier



def ml_loop(clfrs, params, time_splits, label, output_df, \
            outcome_time, date_col):
    '''
    Loops through the classifiers, parameters and train-test pairs and saves
    the results into a pandas dataframe.
    '''
    for name, clfr in clfrs.items():
        print('Working on the {} classifier:'.format(name))
        # unpack parameters list
        param_vals = params[name]
        for p_dict in ParameterGrid(param_vals):
            print('*** with parameters {}'.format(p_dict))
            for t_split in time_splits:

                train = t_split['train']
                test = t_split['test']

                start_date_train = t_split['start_date_train']
                end_date_train = t_split['end_date_train']
                start_date_outcome = t_split['start_date_outcome']
                end_date_outcome = t_split['end_date_outcome']
                start_date_test = t_split['start_date_test']
                end_date_test = t_split['end_date_test']
                print(("Training from {} to {} to predict the" +\
                      " outcome from {} to {} and testing on outcomes from " +\
                      " {} to {}").format(str(start_date_train)[:10],
                                         str(end_date_train)[:10],
                                         str(start_date_outcome)[:10],
                                         str(end_date_outcome)[:10],
                                         str(start_date_test)[:10],
                                         str(end_date_test)[:10]))
                model_clfr = clfr.set_params(**p_dict)

                X_train = train.drop(columns=label)
                y_train = train[label]

                X_test = test.drop(columns=label)
                y_test = test[label]

                model_clfr = clfr.fit(X_train, y_train)

                pred_probs = model_clfr.predict_proba(X_test)[:, 1]

                cutoff_idx = 0.5
                bin_pred = [1 if x > cutoff_idx else 0 for x in pred_probs]
                rec = recall_score(y_test, bin_pred)

                print('**** Model recall is {}'.format(rec))




def set_parameters(type, which_clfrs=None):
    '''
    This function defines which models to run on a test or full set of
    parameters.

    Inputs:
        - chosen_clfrs (dict): dictionary of sci-kit learn classifiers that
            will be ran. By default this function instantiates the following
            classifiers: RandomForest, LogisticRegression, AdaBoost, Bagging,
            SVM, DecisionTree and KNN.
        - type (str): if 'test', returns a small grid of parameters, otherwise
            it returns a larger grid with more parameters

    Returns: the chosen classifiers and parameters

    Source: Adapted from rayid Ghani's
        https://github.com/rayidghani/magicloops/blob/master/simpleloop.py
    '''

    classifiers = {'RandomForest': RandomForestClassifier(),
                   'DecisionTree': DecisionTreeClassifier(),}

    ALL_PARAMS = {
                  'RandomForest': {'n_estimators': [100],
                    'max_depth': [5], 'max_features': ['sqrt'],
                    'min_samples_split': [2], 'n_jobs':[-1]},
                  'DecisionTree': {'criterion': ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': [None,'sqrt','log2'],
                    'min_samples_split': [2,5,10]}}

    if type == 'test':
        parameters = TEST_PARAMS
    else:
        parameters = ALL_PARAMS

    if which_clfrs:
        # filter out chosen classifiers
        if all (key in classifiers for key in which_clfrs):
            clfs = {key: classifiers[key] for key in which_clfrs}
            params = {key: parameters[key] for key in which_clfrs}
            return clfs, params
        else:
            print('Error: Classifier not in list or named incorrectly')
    else:
        return classifiers, parameters
