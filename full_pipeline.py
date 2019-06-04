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
        return

    def create_train_tests(self, outcome_time):
        self.train_test(outcome_time)
        set_count = min(len(self.allegation_sets, self.trr_sets))
        set_list = []
        for i in set_count:
            set_list.append(
                TrainTest(self.allegation_sets[i],
                    self.trr_sets[i]))
        return set_list

class TrainTest:
    #NEED TO ADD METHODS FOR COMPARING REGULAR MODEL TO AUGMENTED MODEL
    def __init__(self, allegation_set, trr_set):
        self.allegation_set = allegation_set
        self.trr_set = trr_set
        self.investigator_set = investigator_set
        self.salary_set = salary_set
        self.officer_df = rd.create_df('officer')
        self.reg = {'train': self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_outcome'))\
             & (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))],
             'test': self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_test')) & \
            (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))]}
        self.aug = {'train': self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_outcome'))\
             & (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))],
             'test': self.officer_df.loc[(
            self.officer_df.appointed_date < trr_set.get('start_date_test')) & \
            (self.officer_df.resignation_date >= \
                trr_set.get('start_date_train'))]}
        self.start_date_train = allegation_set.get('start_date_train')
        self.end_date_train = allegation_set.get('end_date_train')
        self.start_date_outcome = allegation_set.get('start_date_outcome')
        self.end_date_outcome = allegation_set.get('end_date_outcome')
        self.end_date_test = allegation_set.get('end_date_test')


    def add_train_features(self, aug=False):
        ## need to add trr features
        if aug:
            train_df = self.aug['train']
        else:
            train_df = self.reg['train']
        train_df = fg.generate_features(
            train_df, rd.create_df('salary'),
            self.allegation_set.get('train'),
            self.end_train_date)
        train_df = fg.create_sustained_outcome(
            train_df, self.allegation_set.get('train'),
            self.end_train_date)
        train_df = train_df.dropna()
        if aug:
            ## add extra augmented features
            self.aug['train'] = train_df
        else:
            self.reg['train'] = train_df
            pass

    def add_test_features(self, aug=False):
    ## need to add trr features
    ## THIS WILL NEED TO BE EDITED SO CONTINUOUS FEATURES ARE SCALED TO TRAINING SET
        self.reg['test'] = fg.generate_features(
            self.reg['test'], rd.create_df('salary'),
            self.allegation_set.get('test'),
            self.allegation_set.get('end_date_outcome'))
        self.reg['test'] = fg.create_sustained_outcome(
            self.reg['test'], self.allegation_set.get('test'),
            self.allegation_set.get('end_date_outcome'))
        self.reg['test'] = self.reg['test'].dropna()
'''
    def run_loop(self):
        #UPDATE THIS PART LATER TO BE MORE EFFICIENT
        time_splits = self.allegation_set
        time_splits['train'] = self.officer_train
        time_splits['test'] = self.officer_test
        self.output_df = ml.classifier_loop(
            [time_splits], self.grid, self.clfs, self.metric_dict, self.label,
            ['appointed_date', 'resignation_date'], self.metrics, plot=True, save=False)
'''





