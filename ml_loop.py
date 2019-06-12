"""
CAPP 30254: Final project


This file contains the code used to run the models with different parameters 
    and evaluation metrics.

"""

from __future__ import division
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, \
    confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, \
    GridSearchCV, ParameterGrid
from sklearn import preprocessing, svm, metrics, tree, decomposition
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
#from sklearn.neighbors.nearest_centroid import NearestCentroid
#from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta, datetime
import random
from scipy import optimize
import time
import seaborn as sns

#from sklearn.preprocessing import StandardScaler
import itertools
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None
SEED = 84


# KEEP TRACK OF best models and get features importances
################################################################################
# ML loop
################################################################################

def classifier_loop(set_lst, grid, clfrs_to_run, metric_dict, label_lst,
                    metrics, csv_name, plot=False):
    '''
    Loops through the classifiers, parameters and train-test pairs and saves
    the results into a pandas dataframe.

    Inputs:
        - temp_lst (lst): list of temporal splits
        - grid (dict): a parameter grid
        - clfrs_to_run (dict): chosen classifiers to be ran
        - metric_dict (dict): dictionary of metrics with threshold lists
        - label (str): name of label column
        - preds_drop (lst): list of columns that need to be dropped before
            classification
        - metrics (str): choose to get population or regular metrics
        - plot (bool): plots precision recall curve if set to true
        - save (str): saves output df into csv if a filename is provided

    Returns a pandas dataframe with the looping results

    Source: Adapted from rayid Ghani's
        https://github.com/rayidghani/magicloops/blob/master/simpleloop.py
    '''
    # unpack classifier and parameter dictionaries
    clfrs, params = define_ml_models(grid, clfrs_to_run)
    # create output dataframe
    output_df = create_output_df(metric_dict)
    best_mods = {}

    # loop through TrainTest object list
    for st, obj in enumerate(set_lst):
        best_precision = 0
        best_prec_dict = {}
        # unpack the list of parameters and then run loop on combination
        for name, clfr in clfrs.items():
            param_vals = params[name]
            # loops through classifier dictionary
            for i_p, p in enumerate(list(ParameterGrid(param_vals))):
                reg_lst = obj.reg_features
                aug_lst = obj.aug_features
                net_lst = list(set(aug_lst) - set(reg_lst))
                reg_aug_dict = {'regular': reg_lst,
                                'augmented': aug_lst,
                                'network': net_lst}
                for ra, feat_lst in reg_aug_dict.items():
                    # labels: ['firearm_outcome', 'sustained_outcome']
                    for label in label_lst:
                        print('** {} model on outcome {}, TrainTest object {}'.\
                            format(ra, label, st))
                        # get data from dictionary
                        train = obj.train
                        test = obj.test
                        train_start = obj.start_date_train
                        train_end = obj.end_date_train
                        test_start = obj.start_date_test
                        test_end = obj.end_date_test

                        # instantiate classifier
                        cl = clfr.set_params(**p)

                        print('* Working on the {} classifier with parameters \
                            {}.'.format(name, p))
                        print('* Training from {} to {} and testing from {} to \
                            {}.'.format(
                              train_start, train_end, test_start, test_end))

                        # separate into train, test
                        X_train = train[feat_lst]
                        y_train = train[label]
                        X_test = test[feat_lst]
                        y_test = test[label]

                        cl.fit(X_train, y_train)

                        if name == 'SVM':
                            pred_probs = cl.decision_function(X_test)
                        else:
                            pred_probs = cl.predict_proba(X_test)[:, 1]

                        if name == 'Bagging':
                            # https://stackoverflow.com/questions/44333573/feature-importances-bagging-scikit-learn
                            importances = np.mean([
                                          tree.feature_importances_ for tree
                                          in cl.estimators_], axis=0)
                        elif name == 'LogisticRegression':
                            # https://stackoverflow.com/questions/34052115/how-to-find-the-importance-of-the-features-for-a-logistic-regression-model
                            importances = cl.coef_
                        else:
                            importances = cl.feature_importances_
                        # update metrics dataframe
                        update_metrics_df(output_df, y_test, pred_probs, name, 
                                          importances, i_p, p, st, ra, label,
                                          train_start, train_end, test_start,
                                          test_end, metrics)

                        # keeping track of best precision
                        cur_prec_5 = output_df.loc[len(output_df) - 1]\
                            ['precision_at_5']
                        if cur_prec_5 > best_precision:
                            best_precision = cur_prec_5
                            best_prec_dict = {'best_prec': best_precision,
                                             'predicted_probs': pred_probs,
                                             'label': test[label],
                                             'officer_id': test.id}
                        if plot:
                            plot_precision_recall_n(y_test, pred_probs, name,
                                                    plot)
                best_mods[st] = pd.DataFrame.from_dict(best_prec_dict)

    if csv_name:
        output_df.to_csv(csv_name)

    for key, df in best_mods.items():
        df.to_csv(str(key) + '_for_aequitas.csv')

    return output_df

def define_ml_models(type, chosen_clfrs=None):
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
    classifiers = { 'RandomForest': RandomForestClassifier(),
                    'ExtraTrees': ExtraTreesClassifier(),
                    'AdaBoost': AdaBoostClassifier(),
                    'DecisionTree': DecisionTreeClassifier(),
                    'SVM': svm.LinearSVC(),
                    'GradientBoosting': GradientBoostingClassifier(),
                    'Bagging': BaggingClassifier(),
                    'LogisticRegression': LogisticRegressionCV(),
                    'KNN': KNeighborsClassifier()}

    ALL_PARAMS = {
        'RandomForest': {'criterion': ['gini', 'entropy'],
                         'n_estimators': [10, 20, 50, 100],
                         'max_depth': [5, 20, None],
                         'min_samples_split': [10, 100, 500],
                         'n_jobs': [-1],
                         'random_state': [SEED]},
        'SVM': {'penalty': ['l2'],
                'C': [0.1, 1.0, 50.0]},
        'KNN': {'n_neighbors': [5]},
        'DecisionTree': {'criterion': ['gini', 'entropy'],
                   'max_depth': [2, 5],
                   'min_samples_leaf': [10, 500, 1000],
                   'min_samples_split': [5, 50, 100]},
        'AdaBoost': {'n_estimators': [10, 50, 100],
                     'learning_rate': [0.01, 0.1, 0.5, 1],
                     'random_state': [SEED]},
        'Bagging': {'n_estimators': [10, 50, 100],
                    'bootstrap': [True, False],
                    'max_samples': [0.2, 1.0],
                    'n_jobs': [-1],
                    'random_state': [SEED]},
        'GradientBoosting':{'n_estimators': [10, 100, 200],
                            'learning_rate': [0.5, 0.2, 0.05],
                            'max_depth': [2, 5],
                            'min_samples_split': [10, 100, 500],
                            'random_state': [SEED]},
        'LogisticRegression': { 'penalty': ['l2', 'l1'],
                                'Cs': [1, 10, 100],
                                'solver': ['liblinear'],
                                'random_state': [SEED]},
        'ExtraTrees': {'n_estimators': [10, 100, 500],
                       'max_depth': [2, 5, 10],
                       'n_jobs': [-1],
                       'random_state': [SEED]}
                                }

    TEST_PARAMS = {
    'RandomForest':{'n_estimators': [10],
                        'max_depth': [5],
                        'min_samples_split': [5],
                        'n_jobs': [-1],
                        'random_state': [1992]},
    'LogisticRegression': { 'penalty': ['l2'],
                                'Cs': [1]},
    'KNN': {'n_neighbors': [5]},
    'DecisionTree': {}}

    if type == 'test':
        PARAMETERS = TEST_PARAMS
    else:
        PARAMETERS = ALL_PARAMS

    if chosen_clfrs:
        if all (key in classifiers for key in chosen_clfrs):
            clfs = {key: classifiers[key] for key in chosen_clfrs}
            params = {key: PARAMETERS[key] for key in chosen_clfrs}
            return clfs, params
        else:
            print('Error: Classifier not in list or named incorrectly')
            return
    else:
        return classifiers, PARAMETERS

################################################################################
# Evaluation functions
################################################################################

def create_output_df(metric_dict):
    '''
    Creates dataframe to store evaluation results.

    Inputs:
        - metric_dict: (lst) dictionary with evaluation metrics and thresholds.
            Valid metrics are precision, recall, auc-roc, accuracy, f1
    '''

    # minimum columns for output dataframe
    col_lst = ['model', 'importances', 'param_set', 'parameters', 'set', 'type',
               'outcome', 'train_start', 'train_end', 'test_start', 'test_end']
    # dealing with evaluation metrics
    for metric, threshold_lst in metric_dict.items():
        if metric == 'precision' or metric == 'recall' or metric == 'f1' or \
            metric == 'accuracy':
            temp_lst = [metric +'_at_' + str(k) for k in threshold_lst]
            col_lst.extend(temp_lst)
        else:
            col_lst.append(metric)

    output_df = pd.DataFrame(columns=col_lst)

    return output_df

def update_metrics_df(output_df, y_test, pred_probs, name, importances, i_p,
                      parameters, st, ra, label, train_start, train_end,
                      test_start, test_end, metrics):
    '''
    Updates metrics dataframe

    Inputs:
        - output_df: (pandas df) a pandas dataframe with evaluation metrics
        - y_test:
        - pred_probs:
    '''
    # minimum columns to identify model
    result_lst = [name, importances, i_p, parameters, st, ra, label,
                  train_start, train_end, test_start, test_end]

    # list of columns to fill in metrics for
    col_lst = output_df.columns[len(result_lst):]

    # for each column, identify the metric and compute it
    for col in col_lst:
        lst = re.split('_', col)
        metric = lst[0]
        if len(lst) > 2:
            t = int(lst[2])
        if metric == 'precision':
            result = precision_at_k(y_test, pred_probs, t, metrics)
        elif metric == 'recall':
            result = recall_at_k(y_test, pred_probs, t, metrics)
        elif metric == 'f1':
            result = f1_at_k(y_test, pred_probs, t, metrics)
        elif metric == 'accuracy':
            result = accuracy_at_k(y_test, pred_probs, t, metrics)
        elif metric == 'auc-roc':
            result = roc_auc_score(y_test, pred_probs)

        result_lst.append(result)

    # update the row
    output_df.loc[len(output_df)] = result_lst

def plot_roc(name, probs, true, output_type):
    '''
    Plots the ROC curve

    Source: Rayid Ghani
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def gen_binary_at_k(predicted_scores, k, metrics=None):
    '''
    Maps the predicted labels to 0 or 1 given a threshold of k / 100
    '''
    if metrics == 'pop':
        # k is the target percent of population
        #print('population metrics')
        threshold = int(len(predicted_scores) * (k / 100.0))
        binary_preds = [1 if x < threshold else 0 for x in range(\
                len(predicted_scores))]
    else:
        #print('regular metrics')
        threshold = k / 100.0
        binary_preds = [1 if x > threshold else 0 for x in predicted_scores]

    return binary_preds

def f1_at_k(y_true, y_scores, k, metrics):
    '''
    Computes the f1 score at different thresholds (k/100)

    Source: adapted from Rayid Ghani's precision_at_k and recall_at_k functions
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores),
                                                           np.array(y_true))
    preds_at_k = gen_binary_at_k(y_scores_sorted, k, metrics)
    f1 = f1_score(y_true_sorted, preds_at_k)

    return f1

def precision_at_k(y_true, y_scores, k, metrics):
    '''
    Computes the precision score at different thresholds (k/100)

    Source: Rayid Ghani
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores),
                                                           np.array(y_true))
    preds_at_k = gen_binary_at_k(y_scores_sorted, k, metrics)
    precision = precision_score(y_true_sorted, preds_at_k)

    return precision

def recall_at_k(y_true, y_scores, k, metrics):
    '''
    Computes the recall score at different thresholds (k/100)

    Source: Rayid Ghani
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores),
                                                           np.array(y_true))
    preds_at_k = gen_binary_at_k(y_scores_sorted, k, metrics)
    recall = recall_score(y_true_sorted, preds_at_k)

    return recall

def accuracy_at_k(y_true, y_scores, k, metrics):
    '''
    Computes the accuracy score at different thresholds (k/100)

    Source: adapted from Rayid Ghani's precision_at_k and recall_at_k functions
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores),
                                                           np.array(y_true))
    preds_at_k = gen_binary_at_k(y_scores_sorted, k, metrics)
    accuracy = accuracy_score(y_true_sorted, preds_at_k)

    return accuracy

def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    '''
    Plots precision recall

    Source: Rayid Ghani
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = \
        precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


def joint_sort_descending(l1, l2):
    '''
    Sorts two numpy arrays

    Source: Rayid Ghani
        (https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py)
    '''
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]
