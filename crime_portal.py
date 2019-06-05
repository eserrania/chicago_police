import pandas as pd
import numpy as np
from sodapy import Socrata


def read_beat_data():
    crime_beat_quartiles = pd.read_csv('data/crime_beat_quartiles.csv')
    crime_beat_quartiles['crime_month'] = \
        pd.to_datetime(crime_beat_quartiles.crime_month)
    crime_beat_quartiles['crime_month'] = \
        crime_beat_quartiles.crime_month.map(lambda x: x.strftime('%Y-%m'))
    beat = pd.read_csv('data/data_area.csv')
    beat = beat.loc[beat.area_type == 'beat']
    return beat, crime_beat_quartiles

def beat_quartile_trrs(
    officer_df, trr, start_date_train, end_date_train):
    '''
    Adds features indicating average number of trrs per year in beats of
    the first, second, third, and fourth quartiles of crime by month.
    '''
    beat, crime_beat_quartiles = read_beat_data()
    trr = trr.loc[(trr.trr_datetime <= end_date_train) & \
        (trr.trr_datetime >= start_date_train)]
    total_years = (end_date_train - start_date_train) / np.timedelta64(1, 'Y')
    trr['trr_month'] = trr.trr_datetime.map(lambda x: x.strftime('%Y-%m'))
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
        officer_quartiles, , how='left', left_on='id', right_on='officer_id')
    for col_name in ['first_quartile_trrs',
                     'second_quartile_trrs',
                     'third_quartile_trrs',
                     'fourth_quartile_trrs']:
        officer_df[col_name] = officer_df[col_name].fillna(
            officer_df[col_name].mean)
    return officer_df




def beat_quartile_complaints(
    officer_df, merged_allegation, start_date_train, end_date_train):
    '''
    Adds features indicating average number of complaints per year in beats of
    the first, second, third, and fourth quartiles of crime by month.
    '''
    beat, crime_beat_quartiles = read_beat_data()
    merged_allegation = merged_allegation.loc[
        pd.notnull(merged_allegation.beat_id)]
    merged_allegation['beat_id'] = \
        merged_allegation.beat_id.astype('int')
    merged_allegation = \
        merged_allegation.merge(
            beat, left_on='beat_id', right_on='id')
    merged_allegation['name'] = \
        merged_allegation.name.astype('int')
    merged_allegation = merged_allegation.loc[
        (merged_allegation['incident_date'] <= end_date_train),
        (merged_allegation['incident_date'] >= start_date_train)]
    merged_allegation['incident_month'] = \
        merged_allegation.incident_date.map(lambda x: x.strftime('%Y-%m'))
    merged_quartiles = merged_allegation.merge(
        crime_beat_quartiles, 
        left_on=['name', 'incident_month'], 
        right_on=['beat', 'crime_month'])
    officer_quartiles = pd.DataFrame(
        merged_quartiles.groupby(
            ['officer_id', 'quartile'])['allegation_id'].nunique()).reset_index()
    total_years = (end_date_train - start_date_train) / np.timedelta64(1, 'Y')
    officer_quartiles['allegation_id'] = officer_quartiles.id.map(
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
    for col_name in ['first_quartile',
                     'second_quartile',
                     'third_quartile',
                     'fourth_quartile']:
        officer_df[col_name] = officer_df[col_name].fillna(
            officer_df[col_name].mean)
    return officer_df