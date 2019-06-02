import read_data as rd
import train_test as tt
import feature_generation_2 as fg
import ml_loop as ml
import pandas as pd
import numpy as np
import crime_portal as cp




class RawDfs:   
    def __init__(self):
        self.trr_df = rd.create_df('trr_trr')
        self.officer_df = rd.create_df('officer')
        self.allegation_df, self.investigator_df = rd.merge_data(
            rd.create_df('allegation'),
            rd.create_df('officerallegation'),
            rd.create_df('investigator'),
            rd.create_df('investigatorallegation'))
        self.allegation_sets = None
        self.trr_sets = None

    def train_test(self, outcome_time):
        print("Allegation sets")
        print()
        self.allegation_sets = tt.split_sets(
            self.allegation_df,
            outcome_time,
            'incident_date', verbose=True)
        print()
        print()
        print("TRR sets")
        print()
        self.trr_sets = tt.split_sets(self.trr_df,
            outcome_time,
            'trr_datetime', verbose=True)

    def select_sets(self, allegation_set_num, trr_set_num):
        if not self.allegation_sets or not self.trr_sets:
            print("Train test sets have not been created")
            pass
        allegation_set = self.allegation_sets[allegation_set_num]
        trr_set = self.trr_sets[trr_set_num]
        for key in ['start_date_train',
                    'end_date_train',
                    'start_date_outcome',
                    'end_date_outcome',
                    'start_date_test',
                    'end_date_test',
                    'outcome_time']:
            if allegation_set[key] != trr_set[key]:
                print("Allegation " + key + "does not match TRR " + key)
                pass

        else:
            return TrainTest(allegation_set, trr_set)


class TrainTest:
    def __init__(self, allegation_set, trr_set):
        self.allegation_set = allegation_set
        self.trr_set = trr_set
        self.officer_df = rd.create_df('officer')
        self.officer_train = self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_outcome'))\
             & (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))]
        self.officer_test = self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_test')) & \
            (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))]
        self.grid = 'test'
        self.clfs = ['RandomForest']
        self.label = 'sustained_outcome'
        self.metric_dict = {'precision': [100, 5, 20, 50, 80],
               'recall': [100, 5, 20, 50, 80],
               'f1': [5],
               'accuracy': [50]}
        self.metrics = 'pop'
        self.output_df = None

    def add_train_features(self):
    ## need to add trr features
        self.officer_train = fg.generate_features(
            self.officer_train, rd.create_df('salary'),
            self.allegation_set.get('train'),
            self.allegation_set.get('end_date_train'))
        self.officer_train = fg.create_sustained_outcome(
            self.officer_train, self.allegation_set.get('train'),
            self.allegation_set.get('end_date_train'))
        self.officer_train = self.officer_train.dropna()

    def add_test_features(self):
    ## need to add trr features
    ## THIS WILL NEED TO BE EDITED SO CONTINUOUS FEATURES ARE SCALED TO TRAINING SET
        self.officer_test = fg.generate_features(
            self.officer_test, rd.create_df('salary'),
            self.allegation_set.get('test'),
            self.allegation_set.get('end_date_outcome'))
        self.officer_test = fg.create_sustained_outcome(
            self.officer_test, self.allegation_set.get('test'),
            self.allegation_set.get('end_date_outcome'))
        self.officer_test = self.officer_test.dropna()

    def run_loop(self):
        self.output_df = ml.classifier_loop(
            [self.allegation_set], self.grid, self.clfs, self.metric_dict, self.label,
            preds_drop, self.metrics, plot=True, save=False)





