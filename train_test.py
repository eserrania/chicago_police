import pandas as pd
import numpy as np
import math

def split_sets(df, outcome_time, date_col):
    '''
    Given a dataframe, the length of a time period as a numpy Timedelta object, 
    and a date column, returns a list of dictionaries, with each item in the 
    list representing a train, test set, and each dictionary having
    "train" or "test" as keys and the corresponding dataframe with the time
     period as values.

    Creates as many train and test sets as possible where there is at least 
    one year of data to train on for each set.

    Function will also print the corresponding date cutoffs for each set.
    '''
    start_date_train = df[date_col].min()
    final_date = df[date_col].max()
    max_train_years = round(((final_date - \
        2 * (outcome_time)) - start_date_train) / np.timedelta64(1,'Y'))
    count = 0
    set_list = []
    for train_years in range(1, max_train_years + 1):
        print("Sets trained with {} year(s) of data".format(train_years))
        print()
        end_date_test = start_date_train
        increment = 0
        while end_date_test <= final_date: 

            end_date_train = start_date_train + \
                (np.timedelta64(train_years + increment,'Y')) - \
                np.timedelta64(1,'D')
            end_date_outcome = end_date_train + np.timedelta64(1,'D') + \
                outcome_time
            end_date_test = end_date_outcome + outcome_time - \
                np.timedelta64(1,'D')   

            print(
                ("Training set {} is trained from {} to {} to predict the " +\
                "outcome from {} to {} and tested on outcomes in " +\
                "{} to {}").format(
                    str(count), str(start_date_train)[:10],
                    str(end_date_train)[:10],
                    str(end_date_train + np.timedelta64(1,'D'))[:10], 
                    str(end_date_outcome)[:10], 
                    str(end_date_outcome + np.timedelta64(1,'D'))[:10],
                    str(end_date_test)[:10]))
            train_df = df.loc[df[date_col] <= end_date_outcome]
            test_df = df.loc[(df[date_col] > end_date_outcome) & \
                (df[date_col] <= end_date_test)]
            set_dict = {"train": train_df, "test": test_df}
            set_list.append(set_dict)
            increment += 1
            count += 1
        print()
    return set_list

