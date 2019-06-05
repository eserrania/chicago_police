import read_data as rd
import train_test as tt
import feature_generation as fg
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
        set_count = min(len(self.allegation_sets), len(self.trr_sets))
        set_list = []
        for i in range(set_count):
            set_list.append(
                TrainTest(self.allegation_sets[i],
                    self.trr_sets[i]))
        return set_list

class TrainTest:
    #NEED TO ADD METHODS FOR COMPARING REGULAR MODEL TO AUGMENTED MODEL
    def __init__(self, allegation_set, trr_set):
        self.allegation_set = allegation_set
        self.trr_set = trr_set
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
        self.start_date_test = allegation_set.get('start_date_test')
        self.end_date_test = allegation_set.get('end_date_test')
        self.feature_dict = None
        self.feature_list = None


    def add_train_features(self, aug=False):
        ## need to add trr features
        if aug:
            train_df = self.aug['train']
            train_df, feat_dict, feat_lst = \
                fg.generate_features(train_df, self.allegation_set.get('train'),
                                     self.trr_set.get('train'),
                                     self.end_train_date,
                                     augmented=aug)
            self.aug['train'] = train_df
        else:
            train_df = self.reg['train']
            train_df, feat_dict, feat_lst = \
                fg.generate_features(train_df, self.allegation_set.get('train'),
                                     self.trr_set.get('train'),
                                     self.end_train_date)
            self.reg['train'] = train_df

        self.feature_dict = feat_dict
        self.feature_list = feat_lst


    def add_test_features(self, aug=False):
        if aug:
            test_df = self.aug['test']
            test_df = fg.generate_features(train_df,
                                            self.allegation_set.get('test'),
                                            self.trr_set.get('test'),
                                            self.end_train_date,
                                            train_test='test',
                                            self.feature_dict,
                                            augmented=aug)
            self.aug['test'] = test_df

        else:
            test_df = self.aug['test']
            test_df = fg.generate_features(train_df,
                                            self.allegation_set.get('test'),
                                            self.trr_set.get('test'),
                                            self.end_train_date,
                                            train_test='test',
                                            self.feature_dict)
            self.reg['test'] = test_df
