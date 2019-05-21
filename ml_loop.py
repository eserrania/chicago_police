'''
ML loop
'''



def ml_loop(clfrs, params, time_splits, label, output_df, \
            outcome_time, date_col, df):
    '''
    Loops through the classifiers, parameters and train-test pairs and saves
    the results into a pandas dataframe.
    '''
    for name, clfr in clfrs.items():
        print('Working on the {} classifier:'.format(name))
        # unpack parameters list
        param_vals = params[name]
        for p_dict in in ParameterGrid(param_vals):
            for t_split in time_splits:

                train = t_split['train']
                test = t_split['test']

                start_date_train = t_split['start_date_train']
                end_date_train = t_split['end_date_train']
                start_date_outcome = t_split['start_date_outcome']
                end_date_outcome = t_split['end_date_outcome']
                start_date_test = t_split['start_date_test']
                end_date_test = t_split['end_date_test']
                try:
                    print("Training from {} to {} to predict the" +\
                          "outcome from {} to {} and testing on outcomes from " +\
                          "{} to {}").format(str(start_date_train)[:10],
                                             str(end_date_train)[:10],
                                             str(start_date_outcome)[:10],
                                             str(end_date_outcome)[:10],
                                             str(start_date_test)[:10],
                                             str(end_date_test)[:10]
                                             ))
                    model_clfr = clfr.set_params(**p_dict)

                    X_train = train.drop(columns=label)
                    y_train = train[label]

                    X_test = test.drop(columns=label)
                    y_test = test[label]

                    model_clfr = clfr.fit(X_train, y_train)

                    output_df.loc[]
                except:
                    print('ERROR')



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
                   'DecisionTree': DecisionTreeClassifier(),
                   'LogisticRegression': LogisticRegressionCV()}

    ALL_PARAMS = {
                  'RandomForest': {'n_estimators': [100, 10000],
                    'max_depth': [5,50], 'max_features': ['sqrt','log2'],
                    'min_samples_split': [2,10], 'n_jobs':[-1]},
                  'DecisionTree': {'criterion': ['gini', 'entropy'],
                    'max_depth': [1,5,10,20,50,100],
                    'max_features': [None,'sqrt','log2'],
                    'min_samples_split': [2,5,10]},
                  'LogisticRegression': { 'penalty': ['l1'], 'C': [0.01]}
                  }

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
