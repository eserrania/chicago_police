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
        self.victim_df = rd.create_df('victim')
        self.salary_df = rd.create_df('salary')
        self.history_df = rd.create_df('officerhistory')
        self.allegation_sets = []
        self.trr_sets = []
        self.salary_sets = []
        self.investigator_sets = []
        self.victim_sets = []
        self.history_sets = []

    def train_test(self, outcome_time, words=True):
        print("Allegation sets")
        print()
        self.allegation_sets = tt.split_sets(
            self.allegation_df,
            outcome_time,
            'incident_date', verbose=words)
        for all_set in self.allegation_sets:
            inv_set = {}
            vic_set = {}
            history_set = {}
            inv_set['train'] = \
                self.investigator_df.loc[
                    self.investigator_df.allegation_id.isin(
                        list(all_set.get('train').allegation_id))]
            inv_set['test'] = \
                self.investigator_df.loc[
                    self.investigator_df.allegation_id.isin(
                        list(all_set.get('test').allegation_id))]
            vic_set['train'] = \
                self.victim_df.loc[
                    self.victim_df.allegation_id.isin(
                        list(all_set.get('train').allegation_id))]
            vic_set['test'] = \
                self.victim_df.loc[
                    self.victim_df.allegation_id.isin(
                        list(all_set.get('test').allegation_id))]
            history_set['train'] = self.history_df.loc[
                self.history_df.effective_date < \
                    all_set.get('start_date_outcome')] 
            history_set['test'] = self.history_df.loc[
                (self.history_df.end_date >= \
                    all_set.get('start_date_train')) | \
                (self.history_df.end_date.isna())]         
            for date_val in ['start_date_train',
                            'end_date_train',
                            'start_date_outcome',
                            'end_date_outcome',
                            'start_date_test',
                            'end_date_test',
                            'outcome_time']:
                inv_set[date_val] = all_set.get(date_val)
                vic_set[date_val] = all_set.get(date_val)
            self.investigator_sets.append(inv_set)
            self.victim_sets.append(vic_set)
            self.history_sets.append(history_set)
            

        print()
        print()
        print("TRR sets")
        print()
        self.trr_sets = tt.split_sets(
            self.trr_df,
            outcome_time,
            'trr_datetime', verbose=words)
        print()
        print()
        print("Salary sets")
        self.salary_df['salary_date'] = pd.to_datetime(
            ['{}-01-01'.format(round(y)) for y in self.salary_df.year])
        self.salary_sets = tt.split_sets(
            self.salary_df,
            outcome_time,
            'salary_date', verbose=words)
        print()
        print()

        return

    def create_train_tests(self, outcome_time, check_dates=False):
        self.train_test(outcome_time)
        set_count = min(len(self.allegation_sets), len(self.trr_sets),
                        len(self.salary_sets))
        set_list = []
        for i in range(set_count):
            if check_dates:
                for date_val in ['start_date_train',
                                'end_date_train',
                                'start_date_outcome',
                                'end_date_outcome',
                                'start_date_test',
                                'end_date_test',
                                'outcome_time']:
                    print()
                    print(str(i) + str(date_val))
                    for tt_sets in [self.allegation_sets[i],
                                    self.trr_sets[i],
                                    self.salary_sets[i],
                                    self.investigator_sets[i],
                                    self.victim_sets[i]]:
                        print(tt_sets.get(date_val))

            set_list.append(
                TrainTest(self.allegation_sets[i],
                    self.trr_sets[i],
                    self.salary_sets[i],
                    self.investigator_sets[i],
                    self.victim_sets[i],
                    self.history_sets[i]))
        return set_list

class TrainTest:
    #NEED TO ADD METHODS FOR COMPARING REGULAR MODEL TO AUGMENTED MODEL
    def __init__(self, allegation_set, trr_set, salary_set, investigator_set,
        victim_set, history_set):
        self.allegation_set = allegation_set
        self.trr_set = trr_set
        self.salary_set = salary_set
        self.investigator_set = investigator_set
        self.victim_set = victim_set
        self.history_set = history_set
        self.officer_df = rd.create_df('officer')
        self.start_date_train = allegation_set.get('start_date_train')
        self.end_date_train = allegation_set.get('end_date_train')
        self.start_date_outcome = allegation_set.get('start_date_outcome')
        self.end_date_outcome = allegation_set.get('end_date_outcome')
        self.start_date_test = allegation_set.get('start_date_test')
        self.end_date_test = allegation_set.get('end_date_test')
        self.train = self.officer_df.loc[(
            self.officer_df.appointed_date < self.start_date_outcome)\
             & ((self.officer_df.resignation_date >= \
                self.start_date_train) | self.officer_df.resignation_date.isna())]
        self.test = self.officer_df.loc[(
            self.officer_df.appointed_date < self.end_date_test) & \
            (self.officer_df.resignation_date >= \
                self.start_date_test)]
        self.feature_dict = {} 
        self.reg_features = []
        self.aug_features = []


    def add_train_features(self):

        self.train, self.feature_dict, self.reg_features, self.aug_features = \
            fg.generate_features(self.train,
                                 self.allegation_set.get('train'),
                                 self.trr_set.get('train'),
                                 self.victim_set.get('train'),
                                 self.salary_set.get('train'),
                                 self.history_set.get('train'),
                                 self.end_date_train)

        print("Train")
        print(self.trr_set.get('train').groupby('firearm_used')['officer_id'].nunique())
        print(self.train.groupby('used_firearm')['id'].nunique())
        print(self.train.groupby('firearm_outcome')['id'].nunique())
        print()
        print(self.allegation_set.get('train').groupby('final_finding')['officer_id'].nunique())
        print(self.train.groupby('sustained_outcome')['id'].nunique())
        print()



    def add_test_features(self):
        self.test = fg.generate_features(self.test,
                                        self.allegation_set.get('test'),
                                        self.trr_set.get('test'),
                                        self.victim_set.get('test'),
                                        self.salary_set.get('train'),
                                        self.history_set.get('test'),
                                        self.end_date_train,
                                        train_test='test',
                                        feat_dict=self.feature_dict)

        print("Test")
        print(self.trr_set.get('test').groupby('firearm_used')['officer_id'].nunique())
        print(self.test.groupby('used_firearm')['id'].nunique())
        print(self.test.groupby('firearm_outcome')['id'].nunique())
        print()
        print(self.allegation_set.get('test').groupby('final_finding')['officer_id'].nunique())
        print(self.test.groupby('sustained_outcome')['id'].nunique())
        print()
