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
                        'date_cols': ['org_hire_date', 'spp_date', 'start_date']}}

###############################################################################
# Functions (cleaning and raw data exploration)
###############################################################################

def create_df(name):
    '''
    Creates a dataframe given a dictionary that defines the columns to use, the
    data types and date columns.

    Inputs:
        - name: (str) file name, valid names include: 'officer', 'allegation',
          'officer_allegation', 'category', 'victim', 'complainant', 'salary'

    Returns: (df) a pandas dataframe
    '''
    allowed_names = ['officer', 'allegation', 'officerallegation',
                     'allegationcategory', 'victim', 'complainant', 'salary',
                     'geocoordinates']
    try:
        if name == 'geocoordinates':
            df = pd.read_csv('data/' + name + '.csv')
            df['geometry'] = df.apply(lambda x: Point(float(x.x), float(x.y)), axis=1)
            df = gpd.GeoDataFrame(df)
        else:
            dict = data_dict[name]
            df = pd.read_csv('data/data_' + name + '.csv',
                             usecols=dict['use_cols'],
                             parse_dates=dict['date_cols'],
                             dtype=dict['data_types'])
        return df

    except:
        print("ERROR: allowed names are {}".format(allowed_names))
