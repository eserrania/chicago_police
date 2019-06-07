import pandas as pd
import numpy as np
from sodapy import Socrata
from sklearn.preprocessing import MinMaxScaler

def read_beat_data():
    crime_beat_quartiles = pd.read_csv('data/crime_beat_quartiles.csv')
    crime_beat_quartiles['crime_month'] = \
        pd.to_datetime(crime_beat_quartiles.crime_month)
    crime_beat_quartiles['crime_month'] = \
        crime_beat_quartiles.crime_month.map(lambda x: x.strftime('%Y-%m'))
    beat = pd.read_csv('data/data_area.csv')
    beat = beat.loc[beat.area_type == 'beat']
    return beat, crime_beat_quartiles

def beat_quartile_trrs(officer_df, trr_df, end_date_set, feat_dict, train=True):
    '''
    Adds features indicating average number of trrs per year in beats of
    the first, second, third, and fourth quartiles of crime by month.
    '''
    beat, crime_beat_quartiles = read_beat_data()
    trr_df['trr_datetime'] = \
        trr_df.apply(
            lambda x: x['trr_datetime'].tz_localize(None), axis=1)
    trr = trr_df.loc[(trr_df.trr_datetime <= end_date_set)]
    total_years = (end_date_set - np.datetime64('2010-01-01')) \
        / np.timedelta64(365, 'D')
    trr['trr_month'] = trr.trr_datetime.map(lambda x: x.strftime('%Y-%m'))
    trr['beat'] = trr.beat.astype('int')
    merged_quartiles = trr.merge(
        crime_beat_quartiles, 
        left_on=['beat', 'trr_month'], 
        right_on=['beat', 'crime_month'])
    beat_officer_quartiles = pd.DataFrame(merged_quartiles.groupby(
        ['beat', 'officer_id', 'quartile'])['id'].nunique()).reset_index()
    officer_quartiles = pd.DataFrame(beat_officer_quartiles.groupby(
        ['officer_id', 'quartile'])['id'].nunique()).reset_index()
    officer_quartiles['id'] = officer_quartiles.id.map(
        lambda x: x / total_years)
    officer_quartiles = officer_quartiles.pivot_table(
        'id', 'officer_id', 'quartile').fillna(0)
    officer_quartiles.rename(
        columns={'first': 'first_quartile_trrs',
        'second': 'second_quartile_trrs',
        'third': 'third_quartile_trrs',
        'fourth': 'fourth_quartile_trrs'}, inplace=True)
    officer_df = officer_df.merge(
        officer_quartiles, how='left', left_on='id', right_on='officer_id')
    newvars = ['first_quartile_trrs', 'second_quartile_trrs', 
               'third_quartile_trrs', 'fourth_quartile_trrs']

    for col_name in newvars:
        officer_df[col_name] = officer_df[col_name].fillna(
            officer_df[col_name].mean())

    if train:
        for col in newvars:
            scaler = MinMaxScaler()
            officer_df[col] = scaler.fit_transform(np.array(officer_df[col])\
                .reshape(-1, 1))
            feat_dict[col] = scaler

        return officer_df, feat_dict, newvars 

    else:
        for col in newvars:
            scaler = feat_dict[col]
            officer_df[col] = scaler.transform(np.array(officer_df[col])\
                .reshape(-1, 1))

        return officer_df



def beat_quartile_complaints(officer_df, allegation_df, end_date_set,
                             feat_dict, train=True):
    '''
    Adds features indicating average number of complaints per year in beats of
    the first, second, third, and fourth quartiles of crime by month.
    '''
    #end_date_set = end_date_set.tz_localize(None)
    '''
    allegation_df['incident_date'] = \
        allegation_df.apply(
            lambda x: x['incident_date'].tz_localize(None), axis=1)
    '''
    beat, crime_beat_quartiles = read_beat_data()
    allegation_df = allegation_df.loc[
        pd.notnull(allegation_df.beat_id)]
    allegation_df['beat_id'] = \
        allegation_df.beat_id.astype('int')
    allegation_df = \
        allegation_df.merge(
            beat, left_on='beat_id', right_on='id')
    allegation_df['name'] = \
        allegation_df.name.astype('int')
    allegation_df = allegation_df.loc[
        allegation_df['incident_date'] <= end_date_set]
    allegation_df['incident_month'] = \
        allegation_df.incident_date.map(lambda x: x.strftime('%Y-%m'))
    merged_quartiles = allegation_df.merge(
        crime_beat_quartiles, 
        left_on=['name', 'incident_month'], 
        right_on=['beat', 'crime_month'])
    officer_quartiles = pd.DataFrame(
        merged_quartiles.groupby(
            ['officer_id', 'quartile'])['allegation_id'].nunique()).reset_index()
    total_years = (end_date_set - np.datetime64('2010-01-01')) / \
        np.timedelta64(365, 'D')
    officer_quartiles['allegation_id'] = officer_quartiles.allegation_id.map(
        lambda x: x / total_years)
    officer_quartiles = officer_quartiles.pivot_table(
        'allegation_id', 'officer_id', 'quartile').fillna(0)
    officer_quartiles.rename(
        columns={'first': 'first_quartile',
                 'second': 'second_quartile',
                 'third': 'third_quartile', 
                 'fourth': 'fourth_quartile'}, inplace=True)
    officer_df = officer_df.merge(
        officer_quartiles, how='left', left_on='id', right_on='officer_id')

    newvars = ['first_quartile', 'second_quartile', 'third_quartile', 
               'fourth_quartile']

    for col_name in newvars:
        officer_df[col_name] = officer_df[col_name].fillna(
            officer_df[col_name].mean())

    if train:
        for col in newvars:
            scaler = MinMaxScaler()
            officer_df[col] = scaler.fit_transform(np.array(officer_df[col])\
                .reshape(-1, 1))
            feat_dict[col] = scaler

        return officer_df, feat_dict, newvars 

    else:
        for col in newvars:
            scaler = feat_dict[col]
            officer_df[col] = scaler.transform(np.array(officer_df[col])\
                .reshape(-1, 1))

        return officer_df
