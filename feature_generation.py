"""
CAPP 30254: Final project


This file contains the feature generation functions.

"""

import crime_portal as cp
import pandas as pd
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statistics import mode


def generate_features(officer_df, allegation_df, trr_df, victim_df,
                      salary_df, history_df, end_date_set, train_test='train',
                      feat_dict=None):
    '''
    Generates the features needed for the machine learning pipeline,
        differentiating between training and testing sets.
    '''

    officer_df = create_sustained_outcome(officer_df, allegation_df,
                                          end_date_set)
    officer_df = used_firearm(officer_df, trr_df, end_date_set)
    officer_df = create_firearm_outcome(officer_df, trr_df, end_date_set)
    officer_df = create_gender_dummy(officer_df)

    network =  create_coaccusals_network(allegation_df, end_date_set)

    if train_test == 'train':
        features = ['gender', 'age', 'tenure', 'number_complaints',
                    'pct_officer_complaints', 'pct_sustained_complaints',
                    'disciplined_before', 'trr_report_count', 'shooting_count']
        feat_dict = {}

        officer_df, feat_dict, newvars = create_race_dummies(officer_df,
                                                             feat_dict)
        features += newvars

        officer_df, feat_dict = create_age_var(officer_df, end_date_set,
                                               feat_dict)

        officer_df, feat_dict = create_tenure_var(officer_df, end_date_set,
                                                  feat_dict)

        officer_df, feat_dict, newvars = create_rank_dummies(officer_df,
                                                             salary_df,
                                                             end_date_set,
                                                             feat_dict)
        features += newvars

        officer_df, feat_dict, newvars = gen_unit_dummies(officer_df,
                                                          history_df,
                                                          end_date_set,
                                                          feat_dict)
        features += newvars   

        officer_df, feat_dict = gen_trr_counts(officer_df, trr_df, end_date_set,
                                               feat_dict)     

        officer_df, feat_dict = gen_allegation_features(officer_df,
                                                        allegation_df,
                                                        end_date_set, feat_dict)



        officer_df, feat_dict, newvars = gen_victim_features(officer_df,
                                                             allegation_df,
                                                             victim_df, 
                                                             end_date_set,
                                                             feat_dict)
        features += newvars

        officer_df, feat_dict, newvars = cp.beat_quartile_trrs(officer_df,
                                                               trr_df,
                                                               end_date_set,
                                                               feat_dict)
        features += newvars

        officer_df, feat_dict, newvars = \
            cp.beat_quartile_complaints(officer_df, allegation_df, end_date_set,
                                        feat_dict)
        features += newvars

        officer_df, feat_dict, newvars = gen_network_features(officer_df,
                                                              allegation_df,
                                                              network, 
                                                              end_date_set,
                                                              feat_dict)
        features_aug = features[:] + newvars

        return officer_df, feat_dict, features, features_aug



    if train_test == 'test':
        officer_df = create_race_dummies(officer_df, feat_dict, train=False)
        officer_df = create_age_var(officer_df, end_date_set, feat_dict,
                                    train=False)
        officer_df = create_tenure_var(officer_df, end_date_set, feat_dict,
                                       train=False)
        officer_df  = create_rank_dummies(officer_df, salary_df, end_date_set,
                                          feat_dict, train=False)
        officer_df = gen_unit_dummies(officer_df, history_df, end_date_set,
                                      feat_dict, train=False)
        officer_df = gen_trr_counts(officer_df, trr_df, end_date_set, feat_dict,
                                    train=False) 
        officer_df = gen_allegation_features(officer_df, allegation_df,
                                             end_date_set, feat_dict,
                                             train=False)
        officer_df = gen_victim_features(officer_df, allegation_df, victim_df,
                                         end_date_set, feat_dict, train=False)
        officer_df = cp.beat_quartile_trrs(officer_df, trr_df, end_date_set, 
                                        feat_dict, train=False)

        officer_df = cp.beat_quartile_complaints(officer_df, allegation_df, 
                                              end_date_set, feat_dict,
                                              train=False)
        officer_df = gen_network_features(officer_df, allegation_df, network, 
                                          end_date_set, feat_dict, train=False)
        return officer_df


def create_sustained_outcome(officer_df, allegation_df, end_date_set):
    '''
    Given the officers and merged allegation dfs, creates an outcome column
    indicating whether or not the officer had a sustained investigation.
    '''
    outcome_window = allegation_df.loc[
        allegation_df.incident_date > end_date_set]
    sustained = outcome_window.loc[outcome_window.final_finding == "SU"]
    officer_df['sustained_outcome'] = officer_df.apply(
        lambda x: 1 if x['id'] in list(sustained.officer_id) else 0, axis=1)
    return officer_df


def used_firearm(officer_df, trr, end_date_set):
    '''
    Dummy for identifying officers who used a firearm in the training
    period.
    '''
    train_window = trr.loc[
        trr.trr_datetime <= end_date_set]
    firearm_trrs = train_window.loc[train_window.firearm_used]
    officer_df['used_firearm'] = officer_df.apply(
        lambda x: 1 if x['id'] in list(firearm_trrs.officer_id) else 0, axis=1)
    return officer_df


def create_firearm_outcome(officer_df, trr, end_date_set):
    '''
    Given the officers and trr dfs, creates an outcome column
    indicating whether or not the officer used a firearm in the
    outcome window.
    '''
    outcome_window = trr.loc[
        trr.trr_datetime > end_date_set]
    firearm_trrs = outcome_window.loc[outcome_window.firearm_used]
    officer_df['firearm_outcome'] = officer_df.apply(
        lambda x: 1 if x['id'] in list(firearm_trrs.officer_id) else 0, axis=1)
    return officer_df


def create_gender_dummy(officer_df):
    '''
    Given the officers dataframe, convert the gender variable into a dummy.
    '''
    officer_df = officer_df.fillna(value={'gender': mode(officer_df.gender)})
    officer_df['gender'] = np.where(officer_df.gender == 'F', 1 , 0)

    return officer_df


def create_race_dummies(officer_df, feat_dict, train=True):
    '''
    Given the officers dataframe, creates race dummies.
    '''
    officer_df = officer_df.fillna(value={'race': 'unknown'})
    officer_df.race = [race.lower().replace('/', '_').replace(' ', '')
                       for race in officer_df.race]
    if train:
        newvars = []
        values = officer_df.race.unique()
        feat_dict['race'] = values
        officer_df = pd.get_dummies(officer_df, columns=['race'])
        for val in values:
            varname = 'race_{}'.format(val)
            newvars.append(varname)

        return officer_df, feat_dict, newvars

    else:
        for val in feat_dict['race']:
            officer_df['race_{}'.format(val)] = [1 if race == val else 0
                                                 for race in officer_df.race]
        return officer_df


def create_age_var(officer_df, end_date_set, feat_dict, train=True):
    '''
    Given the officers dataframe and the end date of the training data set,
        calculates the officer´s age at that time and scales it.
    '''

    officer_df['age'] = [end_date_set.year - year
                         for year in officer_df.birth_year]

    officer_df = officer_df.fillna(value={'age': np.mean(officer_df.age)})

    if train:
        scaler = MinMaxScaler()
        officer_df['age'] = scaler.fit_transform(\
            np.array(officer_df['age']).reshape(-1, 1))
        feat_dict['age'] = scaler

        return officer_df, feat_dict

    else:
        scaler = feat_dict['age']
        officer_df['age'] = scaler.transform(\
            np.array(officer_df['age']).reshape(-1, 1))

        return officer_df


def create_tenure_var(officer_df, end_date_set, feat_dict, train=True):
    '''
    Given the officers dataframe and the end date of the training data set,
        calculates the officer´s tenure at that timee and scales it.
    '''
    officer_df['tenure'] = [end_date_set.year - date.year
                            for date in officer_df.appointed_date]
    officer_df = officer_df.fillna(value={'tenure': np.mean(officer_df.tenure)})

    if train:
        scaler = MinMaxScaler()
        officer_df['tenure'] = scaler.fit_transform(\
            np.array(officer_df['tenure']).reshape(-1, 1))
        feat_dict['tenure'] = scaler

        return officer_df, feat_dict

    else:
        scaler = feat_dict['tenure']
        officer_df['tenure'] = scaler.transform(\
            np.array(officer_df['tenure']).reshape(-1, 1))

        return officer_df

def create_rank_dummies(officer_df, salary_df, end_date_set, feat_dict,
                        train=True):
    '''
    Given the officers dataframe, their salary history and the end date of the
        training data set creates dummy variables with the officer's rank at
        that time.
    '''
    officer_df = officer_df.drop(columns=['rank'])
    salary = salary_df.loc[salary_df.year == end_date_set.year,
                           ['officer_id', 'rank']]
    officer_df = officer_df.merge(salary, how='left', left_on='id',
                                  right_on='officer_id')
    officer_df = officer_df.fillna(value={'rank': 'unknown'})

    if train:
        newvars = []
        values = officer_df['rank'].unique()
        feat_dict['rank'] = values
        officer_df = pd.get_dummies(officer_df, columns=['rank'])
        for val in values:
            varname = 'rank_{}'.format(val)
            newvars.append(varname)
        return officer_df, feat_dict, newvars

    else:
        for val in feat_dict['rank']:
            officer_df['rank_{}'.format(val)] = [1 if rank == val else 0
                                                 for rank in officer_df['rank']]
        return officer_df

def gen_unit_dummies(officer_df, history_df, end_date_set, feat_dict,
                     train=True):
    '''
    Generate dummies indicating the officer's unit at the end of the end date
        of the data set.
    '''

    history = history_df[(history_df.effective_date <= end_date_set)]
    history = history.groupby('officer_id')['effective_date',
                                            'unit_id'].max().reset_index()
    history.drop(columns=['effective_date'], inplace=True)
    officer_df = officer_df.merge(history, how='left', left_on='id',
                                  right_on='officer_id')
    if train:
        newvars = []
        values = officer_df['unit_id'].unique()
        feat_dict['unit_id'] = values
        officer_df = pd.get_dummies(officer_df, columns=['unit_id'])
        for val in values:
            varname = 'unit_id_{}'.format(val)
            newvars.append(varname)
        return officer_df, feat_dict, newvars

    else:
        for val in feat_dict['unit_id']:
            officer_df['unit_id_{}'.format(val)] = [1 if unit == val else 0
                                                    for unit in 
                                                    officer_df['unit_id']]
        return officer_df



def gen_trr_counts(officer_df, trr_df, end_date_set, feat_dict, train=True):
    '''
    Generate a count of the total trr reports and times an officer used a
        firearm.
    '''
    trr = trr_df[trr_df.trr_datetime <= end_date_set]
    trr = trr[trr.officer_id.isin(officer_df.id.unique())]

    trr_count = trr.groupby('officer_id').size().to_frame().reset_index()\
        .rename(columns={0: 'trr_report_count'})

    officer_df = officer_df.merge(trr_count, how='left', left_on='id',
                                  right_on='officer_id')
    print(officer_df.trr_report_count.describe())


    firearm = trr[trr.firearm_used == True].groupby('officer_id').size()\
        .to_frame().reset_index().rename(columns={0: 'shooting_count'})
    officer_df = officer_df.merge(firearm, how='left', left_on='id',
                                  right_on='officer_id')

    officer_df = officer_df.fillna(value={'trr_report_count': 0,
            'shooting_count': 0})
    print(officer_df.trr_report_count.describe())


    if train:
        scaler_1 = MinMaxScaler()
        scaler_2 = MinMaxScaler()

        officer_df['trr_report_count'] = scaler_1.fit_transform(\
            np.array(officer_df['trr_report_count']).reshape(-1, 1))

        officer_df['shooting_count'] = scaler_2.fit_transform(\
            np.array(officer_df['shooting_count']).reshape(-1, 1))
        
        feat_dict['trr_report_count'] = scaler_1
        feat_dict['shooting_count'] = scaler_2

        return officer_df, feat_dict

    else:
        scaler_1 = feat_dict['trr_report_count']
        officer_df['trr_report_count'] = scaler_1.transform(\
            np.array(officer_df['trr_report_count']).reshape(-1, 1))

        scaler_2 = feat_dict['shooting_count']
        officer_df['shooting_count'] = scaler_2.transform(\
            np.array(officer_df['shooting_count']).reshape(-1, 1))

        return officer_df



def gen_allegation_features(officer_df, allegation_df, end_date_set, feat_dict,
                            train=True):
    '''
    Given the officers dataframe and the their complaint history, creates
        variables with their complaint counts, percentage of officer complaints,
        and percentage of sustained complaints.
    '''

    allegation_df = allegation_df[allegation_df.incident_date <= end_date_set]
    allegation_df = allegation_df[allegation_df['officer_id']\
        .isin(officer_df.id.unique())]
    allegation_df['disciplined'] = np.where(allegation_df.disciplined == 'true',
                                            True, False)
    allegation_df = allegation_df.fillna(value={'disciplined': False})


    disciplined = allegation_df[allegation_df.disciplined].officer_id.unique()

    officer_df['disciplined_before'] = [1 if oid in disciplined else 0 
                                        for oid in officer_df.id]
    officer_df['number_complaints'] = 0
    officer_df['pct_officer_complaints'] = 0.0
    officer_df['pct_sustained_complaints'] = 0.0
    off_complaints = allegation_df[allegation_df.is_officer_complaint == True]\
        .officer_id.value_counts()
    sustained = allegation_df[allegation_df.final_finding == 'SU'].officer_id\
        .value_counts()
    count = allegation_df.officer_id.value_counts()


    for oid in allegation_df.officer_id.unique():
        officer_df.loc[officer_df.id == oid, 'number_complaints'] = count[oid]

        if oid in off_complaints.index:
            officer_df.loc[officer_df.id == oid, 'pct_officer_complaints'] = \
            off_complaints[oid] / count[oid]

        if oid in sustained.index:
            officer_df.loc[officer_df.id == oid, 'pct_sustained_complaints'] = \
            sustained[oid] / count[oid]

    if train:
        scaler = MinMaxScaler()
        officer_df['number_complaints'] = scaler.fit_transform(\
            np.array(officer_df['number_complaints']).reshape(-1, 1))
        feat_dict['number_complaints'] = scaler

        return officer_df, feat_dict

    else:
        scaler = feat_dict['number_complaints']
        officer_df['number_complaints'] = scaler.transform(\
            np.array(officer_df['number_complaints']).reshape(-1, 1))
        return officer_df

def gen_victim_features(officer_df, allegation_df, victim_df, end_date_set,
                        feat_dict, train=True):
    '''
    Generate a victim count per officer and scale it, as well as percentages of 
        victims per race. 
    '''
    officer_df['victim_count'] = 0
    officer_df['black_count'] = 0
    officer_df['white_count'] = 0
    officer_df['api_count'] = 0
    officer_df['hispanic_count'] = 0
    officer_df['pct_white_victims'] = 0.0
    officer_df['pct_black_victims'] = 0.0
    officer_df['pct_api_victims'] = 0.0
    officer_df['pct_hispanic_victims'] = 0.0

    allegation_df = allegation_df[allegation_df.incident_date <= end_date_set]

    allegs = allegation_df[allegation_df.officer_id.isin(officer_df.id)].crid
    victim_filter = victim_df[victim_df.allegation_id.isin(allegs)]

    for aid in victim_filter.allegation_id.unique():
        victims = victim_filter[victim_filter.allegation_id == aid]

        cnt = len(victims)
        if cnt > 0:
            white = len(victims[victims.race == 'White'])
            black = len(victims[victims.race == 'Black'])
            hisp = len(victims[victims.race == 'Hispanic'])
            api = len(victims[victims.race == 'Asian/Pacific Islander'])
            officers = allegation_df[allegation_df.crid == aid].officer_id
            officer_df.loc[officer_df.id.isin(officers), 'victim_count'] += cnt
            officer_df.loc[officer_df.id.isin(officers), 'white_count'] += \
                white
            officer_df.loc[officer_df.id.isin(officers), 'black_count'] += \
                black
            officer_df.loc[officer_df.id.isin(officers), 'api_count'] += api
            officer_df.loc[officer_df.id.isin(officers), 'hispanic_count'] += \
                hisp

    have_victims = officer_df.victim_count > 0
    officer_df.loc[have_victims, 'pct_white_victims'] = \
        officer_df[have_victims].white_count /\
        officer_df[have_victims].victim_count

    officer_df.loc[have_victims, 'pct_black_victims'] = \
        officer_df[have_victims].black_count /\
        officer_df[have_victims].victim_count

    officer_df.loc[have_victims, 'pct_api_victims'] = \
        officer_df[have_victims].api_count /\
        officer_df[have_victims].victim_count

    officer_df.loc[have_victims, 'pct_hispanic_victims'] = \
        officer_df[have_victims].hispanic_count /\
        officer_df[have_victims].victim_count


    officer_df.drop(columns=['black_count', 'white_count', 'api_count',
                             'hispanic_count'], inplace=True)

    if train:
        scaler = MinMaxScaler()
        officer_df['victim_count'] = scaler.fit_transform(\
            np.array(officer_df['victim_count']).reshape(-1, 1))
        feat_dict['victim_count'] = scaler

        return officer_df, feat_dict, ['victim_count', 'pct_white_victims',
                                        'pct_black_victims', 'pct_api_victims',
                                        'pct_hispanic_victims']

    else:
        scaler = feat_dict['victim_count']
        officer_df['victim_count'] = scaler.transform(\
            np.array(officer_df['victim_count']).reshape(-1, 1))

        return officer_df






def create_coaccusals_network(allegation_df, end_date_set):
    '''
    Create a network of officers that have been coaccused together in the past.
    '''

    G = nx.Graph()

    allegations = allegation_df[allegation_df.incident_date \
                                <= end_date_set].crid

    for aid in allegations.unique():
        officers = allegation_df[allegation_df.crid == aid]
        if len(officers) > 1:
            oids = officers.officer_id
            n = 0
            for oid in oids:
                for oid_2 in oids[n:]:
                    if oid != oid_2:
                        if (oid, oid_2) in G.edges():
                            G.edges[oid, oid_2]['count'] += 1
                            G.edges[oid, oid_2]['weight'] = 1 / \
                                G.edges[oid, oid_2]['count']
                        else:
                            G.add_edge(oid, oid_2, count=1, weight=1)
                n += 1
    return G


def add_investigators_network(network, investigators_df, allegation_df,
                              end_date_set):
    '''
    Add investigators to the officer network.
    '''

    investigators_df = investigators_df[investigators_df.officer_id.notnull()]
    allegations = allegation_df[allegation_df.incident_date \
                                <= end_date_set].crid

    for aid in allegations.unique():
        investigator = investigators_df[investigators_df.allegation_id == aid]
        officers = allegation_df[allegation_df.crid == aid]
        if (len(investigator) > 0) & (len(officers) > 0):
            inv_id = investigator.investigator_id
            off_id = officers.officer_id

            for iid in inv_id:
                for oid in off_id:
                    if (iid, oid) in network.edges():
                        network.edges[iid, oid]['count'] += 1
                        network.edges[iid, oid]['weight'] = 1 / \
                            network.edges[iid, oid]['count']
                    else:
                        network.add_edge(iid, oid, count=1, weight=1)

    return network



def create_network(allegation_df, investigators_df, history_df, end_date_set):
    '''
    Create a network of coaccusals and investigations among police officers.
    '''
    nw = create_coaccusals_network(allegation_df, end_date_set)
    nw = add_investigators_network(nw, investigators_df, allegation_df,
                                   end_date_set)
    return nw


def gen_network_features(officer_df, allegation_df, network, end_date_set,
                         feat_dict, train=True):
    '''
    Generate features from a previously created network of police officers.
    '''

    firearm_oid = officer_df[officer_df.used_firearm == 1].id.unique()
    sustained_oid = allegation_df[allegation_df.final_finding == "SU"]\
        .officer_id.unique()

    officer_df['shortest_path_shooting_officer'] = None
    officer_df['shortest_path_sustained_officer'] = None
    officer_df['shortest_path_below_four_shooting'] = None
    officer_df['shortest_path_below_four_sustained'] = None


    for oid in officer_df.id:
        below_four_firearm = 0
        below_four_sustained = 0
        shortest_firearm = None
        shortest_sustained = None

        if oid in network.nodes:

            for foid in firearm_oid:

                if foid in network.nodes:
                    if nx.has_path(network, oid, foid):
                        length = nx.shortest_path_length(network, oid, foid)

                        if length < 4:
                            below_four_firearm += 1

                        if shortest_firearm:
                            if length < shortest_firearm:
                                shortest_firearm = length
                        else:
                            shortest_firearm = length


            if shortest_firearm:
                officer_df.loc[officer_df.id == oid,
                               'shortest_path_shooting_officer'] = \
                                round(shortest_firearm)

            else:
                officer_df.loc[officer_df.id == oid,
                               'shortest_path_shooting_officer'] = 'no_path'

            for soid in sustained_oid:
                if soid in network.nodes:
                    if nx.has_path(network, oid, soid):
                        length = nx.shortest_path_length(network, oid, soid)

                        if length < 4:
                            below_four_sustained += 1

                        if shortest_sustained:
                            if length < shortest_sustained:
                                shortest_sustained = length

                        else:
                            shortest_sustained = length

            if shortest_sustained:
                officer_df.loc[officer_df.id == oid,
                               'shortest_path_sustained_officer'] = \
                               round(shortest_sustained)

            else:
                officer_df.loc[officer_df.id == oid,
                               'shortest_path_sustained_officer'] = 'no_path'


        else:
            officer_df.loc[officer_df.id == oid,
                           'shortest_path_shooting_officer'] = 'no_path'

            officer_df.loc[officer_df.id == oid,
                           'shortest_path_sustained_officer'] = 'no_path'

        officer_df.loc[officer_df.id == oid,
                       'shortest_path_below_four_shooting'] = \
                       below_four_firearm

        officer_df.loc[officer_df.id == oid,
                       'shortest_path_below_four_sustained'] = \
                       below_four_sustained

    if train:
        newvars = ['shortest_path_below_four_sustained',
                   'shortest_path_below_four_shooting']

        values_1 = officer_df['shortest_path_shooting_officer'].unique()
        values_2 = officer_df['shortest_path_sustained_officer'].unique()
        feat_dict['shortest_path_shooting_officer'] = values_1
        feat_dict['shortest_path_sustained_officer'] = values_2


        for val in officer_df['shortest_path_shooting_officer'].unique():
            varname = 'shortest_path_shooting_officer_{}'.format(val)
            newvars.append(varname)

        for val in officer_df['shortest_path_sustained_officer'].unique():
            varname = 'shortest_path_sustained_officer_{}'.format(val)
            newvars.append(varname)

        officer_df = pd.get_dummies(officer_df,
                                    columns=['shortest_path_shooting_officer',
                                             'shortest_path_sustained_officer'])

        scaler_1 = MinMaxScaler()
        scaler_2 = MinMaxScaler()

        officer_df['shortest_path_below_four_shooting'] = \
            scaler_1.fit_transform(np.array(\
                officer_df['shortest_path_below_four_shooting']).reshape(-1,
                                                                         1))

        feat_dict['shortest_path_below_four_shooting'] = scaler_1

        officer_df['shortest_path_below_four_sustained'] = \
            scaler_2.fit_transform(np.array(\
                officer_df['shortest_path_below_four_sustained']).reshape(-1,
                                                                          1))

        feat_dict['shortest_path_below_four_sustained'] = scaler_2

        return officer_df, feat_dict, newvars

    else:
        for val in feat_dict['shortest_path_sustained_officer']:
            varname = 'shortest_path_sustained_officer_{}'.format(val)

            officer_df[varname] = [1 if x == val else 0 for x in
                                   officer_df.shortest_path_sustained_officer]

        for val in feat_dict['shortest_path_shooting_officer']:
            varname = 'shortest_path_shooting_officer_{}'.format(val)

            officer_df[varname] = [1 if x == val else 0 for x in
                                   officer_df.shortest_path_shooting_officer]

        scaler_1 = feat_dict['shortest_path_below_four_shooting']
        scaler_2 = feat_dict['shortest_path_below_four_sustained']

        officer_df['shortest_path_below_four_shooting'] = scaler_1.transform(\
            np.array(officer_df['shortest_path_below_four_shooting']).\
                reshape(-1, 1))

        officer_df['shortest_path_below_four_sustained'] = scaler_2.transform(\
            np.array(officer_df['shortest_path_below_four_sustained']).\
                reshape(-1, 1))

        return officer_df
