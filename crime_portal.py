import pandas as pd
import numpy as np
from sodapy import Socrata


def find_crime_rates(df):
    '''
    Given a dataframe, adds a column representing the total number
    of crimes which occurred in that district and month based on API
    calls to the Chicago Data Portal.
    Note: This function makes individual API calls to the Chicago
    Data Portal and can take at least 15 minutes to complete.
    Inputs:
        df (dataframe): Any dataframe with unit_name and incident_date
    Outputs:
        Returns a dataframe with a crimes column representing the
        number of crimes that occurred in that district in that month.
    '''
    df = df.loc[df.incident_date >= pd.to_datetime('2001-02-01')]
    df = df.loc[df.unit_name.notna()]
    df['unit_name'] = df.unit_name.astype('int')
    df = df.loc[df.unit_name <= 25]
    df['year'] = df.incident_date.apply(lambda x: x.year)
    df['month'] = df.incident_date.apply(lambda x: x.month)
    unit_dates = df[['unit_name', 'year', 'month']].drop_duplicates()
    unit_dates['crimes'] = np.nan
    client = Socrata("data.cityofchicago.org", '6sr95dE6LHGM6Ga2Z2kOU2OfL')
    for index, row in unit_dates.iterrows():
        year = str(int(row['year']))
        month = ("0" + str(int(row['month'])))[-2:]
        if int(row['month']) == 12:
            next_month = '01'
            next_year = str(int(row['year']) + 1)
        else:
            next_month = ("0" + str(int(row['month'] + 1)))[-2:]
            next_year = year

        district = ("00" + str(int(row['unit_name'])))[-3:]
        unit_dates.at[index, 'crimes'] = client.get("6zsd-86xi",
                                                    select='COUNT(*)',
                                                    where='date >= \'{}-{}-01\'\
                                                    and date < \'{}-{}-01\' and \
                                                    district = \'{}\''.format(
                                                        year, month, next_year, next_month,
                                                        district))[0].get(
                                                            'COUNT', 0)
    df = df.merge(unit_dates, how='left', on=['unit_name', 'month', 'year'])
    df = df.drop(columns=['year', 'month'])

    return df