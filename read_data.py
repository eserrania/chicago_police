'''
Read data
'''

from shapely.geometry import Polygon
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import numpy as np

###############################################################################
# Data dictionaries
###############################################################################
data_dict = {'officer': {'use_cols': ['id', 'gender', 'race', 'appointed_date',
                                      'rank', 'active', 'birth_year',
                                      'resignation_date', 'current_badge',
                                      'current_salary', 'last_unit_id'],
                         'data_types': {'id': 'str',
                                        'gender': 'str',
                                        'race': 'str',
                                        'appointed_date': 'str',
                                        'rank': 'str',
                                        'active': 'str',
                                        'birth_year': 'float',
                                        'resignation_date': 'str',
                                        'current_badge': 'str',
                                        'current_salary': 'float',
                                        'last_unit_id': 'str'},
                          'date_cols': ['appointed_date',
                                        'resignation_date']},
             'allegation': {'use_cols': ['crid', 'summary', 'add2', 'city',
                                         'incident_date', 'point', 'source',
                                         'beat_id', 'is_officer_complaint',
                                         'add1', 'location', 'first_end_date',
                                         'first_start_date', 'subjects'],
                            'data_types': {'crid': 'str',
                                           'summary': 'str',
                                           'add2': 'str',
                                           'city': 'str',
                                           'incident_date': 'str',
                                           'point': 'str',
                                           'source': 'str',
                                           'beat_id': 'str',
                                           'is_officer_complaint': 'bool',
                                           'add1': 'str',
                                           'location': 'str',
                                           'first_end_date':'str',
                                           'first_start_date': 'str',
                                           'subjects': 'str'},
                            'date_cols': ['incident_date', 'first_end_date',
                                          'first_start_date']},
             'officerallegation': {'use_cols': ['id', 'start_date', 'end_date',
                                                 'recc_finding',
                                                 'recc_outcome', 'final_finding',
                                                 'final_outcome_class',
                                                 'allegation_category_id', 'officer_id',
                                                 'disciplined', 'allegation_id'],
                                    'data_types': {'id': 'str',
                                                   'start_date': 'str',
                                                   'end_date': 'str',
                                                   'recc_finding': 'str',
                                                   'recc_outcome': 'str',
                                                   'final_finding': 'str',
                                                   'final_outcome': 'str',
                                                   'allegation_category_id': 'str',
                                                   'officer_id': 'str',
                                                   'disciplined': 'str',
                                                   'allegation_id': 'str'},
                                    'date_cols': ['start_date', 'end_date']},
             'allegationcategory': {'use_cols': ['id', 'category_code', 'category',
                                       'allegation_name', 'citizen_dept'],
                          'data_types': {'id': 'str',
                                         'category_code': 'str',
                                         'category': 'str',
                                         'allegation_name': 'str'},
                          'date_cols': []},
             'victim': {'use_cols': ['id', 'gender', 'race', 'age',
                                     'birth_year', 'allegation_id'],
                        'data_types': {'id': 'str',
                                         'gender': 'str',
                                         'race': 'str',
                                         'age': 'str',
                                         'birth_year': 'float',
                                         'allegation_id': 'str'},
                        'date_cols': []},
             'complainant': {'use_cols': ['id', 'gender', 'race', 'age',
                                     'birth_year', 'allegation_id'],
                             'data_types': {'id': 'str',
                                            'gender': 'str',
                                            'race': 'str',
                                            'age': 'str',
                                            'birth_year': 'float',
                                            'allegation_id': 'str'},
                             'date_cols': []},
             'salary': {'use_cols': ['id', 'pay_grade', 'rank', 'salary',
                                     'employee_status', 'org_hire_date',
                                     'spp_date', 'start_date', 'year',
                                     'age_at_hire', 'officer_id',
                                     'rank_changed'],
                        'data_types': {'id': 'str',
                                       'pay_grade': 'str',
                                       'rank': 'str',
                                       'salary': 'float',
                                       'employee_status': 'str',
                                       'org_hire_date': 'str',
                                       'spp_date': 'str',
                                       'start_date': 'str',
                                       'year': 'float',
                                       'age_at_hire': 'float',
                                       'officer_id': 'str',
                                       'rank_changed': 'bool'},
                        'date_cols': ['org_hire_date', 'spp_date', 'start_date']},
                'trr_trr': {'use_cols': ['id', 'beat', 'block', 'direction',
                                         'street', 'location',
                                         'trr_datetime', 'indoor_or_outdoor',
                                         'lighting_condition',
                                         'weather_condition',
                                         'notify_OEMC',
                                         'notify_district_sergeant',
                                         'notify_OP_command',
                                         'notify_DET_division',
                                         'number_of_weapons_discharged',
                                         'party_fired_first',
                                         'location_recode',
                                         'taser',
                                         'total_number_of_shots',
                                         'firearm_used',
                                         'number_of_officers_using_firearm',
                                         'officer_assigned_beat',
                                         'officer_on_duty',
                                         'officer_in_uniform',
                                         'officer_injured',
                                         'officer_rank', 'subject_id',
                                         'subject_armed', 'subject_injured',
                                         'subject_alleged_injury', 'subject_age',
                                         'subject_birth_year',
                                         'subject_gender', 'subject_race',
                                         'officer_id', 'officer_unit_id',
                                         'officer_unit_detail_id', 'point'],
                            'data_types': {'id': 'str', 'beat': 'str',
                                             'block': 'str', 'direction': 'str',
                                             'street': 'str', 'location': 'str',
                                             'trr_datetime': 'str',
                                             'indoor_or_outdoor': 'str',
                                             'lighting_condition': 'str',
                                             'weather_condition': 'str',
                                             'notify_OEMC': 'bool',
                                             'notify_district_sergeant': 'bool',
                                             'notify_OP_command': 'bool',
                                             'notify_DET_division': 'bool',
                                             'number_of_weapons_discharged': 'float',
                                             'party_fired_first': 'str',
                                             'location_recode': 'str',
                                             'taser': 'bool',
                                             'total_number_of_shots': 'float',
                                             'firearm_used': 'bool',
                                             'number_of_officers_using_firearm': 'float',
                                             'officer_assigned_beat': 'str',
                                             'officer_on_duty': 'bool',
                                             'officer_in_uniform': 'bool',
                                             'officer_injured': 'bool',
                                             'officer_rank': 'str',
                                             'subject_id': 'str',
                                             'subject_armed': 'str',
                                             'subject_injured': 'str',
                                             'subject_alleged_injury': 'str',
                                             'subject_age': 'float',
                                             'subject_birth_year': 'float',
                                             'subject_gender': 'str',
                                             'subject_race': 'str',
                                             'officer_id': 'str',
                                             'officer_unit_id': 'str',
                                             'officer_unit_detail_id': 'str',
                                             'point': 'str'},
                            'date_cols': ['trr_datetime']},
                'investigator': {'use_cols': ['id', 'appointed_date',
                                             'first_name', 'last_name',
                                             'middle_initial', 'officer_id',
                                             'suffix_name', 'gender', 'race'],
                                'data_types': {'id': 'str',
                                               'appointed_date': 'str',
                                               'first_name': 'str',
                                               'last_name': 'str',
                                               'middle_initial': 'str',
                                               'officer_id': 'str',
                                               'suffix_name': 'str',
                                               'gender': 'str',
                                               'race': 'str'},
                                'date_cols': ['appointed_date']},
                'investigatorallegation': {'use_cols': ['id', 'current_star',
                                                        'current_rank',
                                                        'current_unit_id',
                                                        'investigator_id',
                                                        'investigator_type',
                                                        'allegation_id'],
                                            'data_types': {'id': 'str',
                                                           'current_star': 'str',
                                                           'current_rank': 'str',
                                                           'current_unit_id': 'str',
                                                           'investigator_id': 'str',
                                                           'investigator_type': 'str',
                                                           'allegation_id': 'str'},
                                            'date_cols': []},
                'officerhistory': {'use_cols': ['id', 'effective_date',
                                                'end_date', 'officer_id',
                                                'unit_id'],
                                    'data_types': {'id': 'str',
                                                   'effective_date': 'str',
                                                   'end_date': 'str',
                                                   'officer_id': 'str',
                                                   'unit_id': 'str'},
                                    'date_cols': ['effective_date', 'end_date']}
                }

###############################################################################
# Functions (cleaning and raw data exploration)
###############################################################################

def create_df(name):
    '''
    Creates a dataframe given a dictionary that defines the columns to use, the
    data types and date columns.

    Inputs:
        - name: (str) file name, valid names include: 'officer', 'allegation',
          'officer_allegation', 'category', 'victim', 'complainant', 'salary',
          'trr_trr'

    Returns: (df) a pandas dataframe
    '''
    allowed_names = ['officer', 'allegation', 'officerallegation',
                     'allegationcategory', 'victim', 'complainant', 'salary',
                     'geocoordinates', 'trr_trr']
    try:
        if name == 'geocoordinates':
            df = pd.read_csv('data/' + name + '.csv')
            df['geometry'] = df.apply(lambda x: Point(float(x.x), float(x.y)), axis=1)
            df = gpd.GeoDataFrame(df)
        else:
            dict = data_dict[name]
            prefix = 'data/data_'
            if name =='trr_trr':
                prefix = 'data/'
            df = pd.read_csv(prefix + name + '.csv',
                             usecols=dict['use_cols'],
                             parse_dates=dict['date_cols'],
                             dtype=dict['data_types'])
        return df

    except Exception as e:
        print("ERROR:", e)
        #print("ERROR: allowed names are {}".format(allowed_names))

def merge_data(allegation_df, officerallegation_df, investigator_df,
               investigatorallegation_df):
    '''
    Merge allegation with officer allegation data and investigator with
    investigator allegation.
    '''
    allegationm_df = allegation_df.merge(officerallegation_df, left_on='crid', right_on='allegation_id')
    investigatorm_df = investigator_df.merge(investigatorallegation_df, left_on='id', right_on='investigator_id')

    return (allegationm_df, investigatorm_df)
