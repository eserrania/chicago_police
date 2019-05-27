import pandas as pd
import numpy as np
from sodapy import Socrata


def beat_quartile_complaints(officer_df, merged_allegation, end_date_train):
    '''
    Adds features indicating number of complaints in beats of the first,
    second, third, and fourth quartiles of crime by month.
    '''
    crime_beat_quartiles = pd.read_csv('data/crime_beat_quartiles.csv')
    crime_beat_quartiles['crime_month'] = \
        pd.to_datetime(crime_beat_quartiles.crime_month)
    crime_beat_quartiles['crime_month'] = \
        crime_beat_quartiles.crime_month.map(lambda x: x.strftime('%Y-%m'))
    beat = pd.read_csv('data/data_area.csv')
    beat = beat.loc[beat.area_type == 'beat']
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
        merged_allegation['incident_date'] <= end_date_train]
    merged_allegation['incident_month'] = \
        merged_allegation.incident_date.map(lambda x: x.strftime('%Y-%m'))
    merged_quartiles = merged_allegation.merge(
        crime_beat_quartiles, 
        left_on=['name', 'incident_month'], 
        right_on=['beat', 'crime_month'])
    officer_quartiles = pd.DataFrame(
        merged_quartiles.groupby(
            ['officer_id', 'quartile'])['allegation_id'].nunique()).reset_index()
    officer_quartiles = officer_quartiles.pivot_table(
        'allegation_id', 'officer_id', 'quartile').fillna(0)
    officer_quartiles.rename(
        columns={'first': 'first_quartile',
                 'second': 'second_quartile',
                 'third': 'third_quartile', 
                 'fourth': 'fourth_quartile'}, inplace=True)



    return officer_df.merge(
        officer_quartiles, left_on='id', right_on='officer_id')