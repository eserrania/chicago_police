import pandas as 
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt


def create_gender_dummy(officer_df):
    '''
    Transforms gender into a dummy variable for each value of the specified variables.

    Input:
        df: (pandas.DataFrame) a data frame

    Returns:
        pandas.DataFrame
    '''
    officer_df.gender = np.where(officer_df.gender == 'F', 1 , 0)
    
    return officer_df


def create_race_dummies(officer_df):
    '''
    '''
    officer_df.race = [race.lower().replace('/', '_').replace(' ', '') 
                       for race in officer_df.race]
    officer_df = pd.get_dummies(officer_df, columns=['race'])
    return officer_df


def create_age_dummies(officer_df, split_date, bins=4):
    '''
    '''
            
    officer_df['age'] = pd.cut(split_date.year - officer_df.birth_year, bins)
    officer_df = pd.get_dummies(officer_df, columns=['age'])
    return officer_df


def create_tenure_dummies(officer_df, split_date, bins=4):
    '''
    '''
    officer_df['tenure'] = [math.floor(((split_date - date).days) / 365.25) 
                            for date in officers.appointed_date]
    officer_df['tenure'] = pd.cut(officer_df.tenure, bins)
    officer_df = pd.get_dummies(officer_df, columns=['tenure'])
    return officer_df


def create_rank_dummies(officer_df, salary_df, split_date, bins=4):
    '''
    '''
    officer_df = officer_df.drop(columns=['rank'])
    salary_df = salary_df[[ryear <= split_date.year for ryear in salary_df.year]]
    salary_df = salary_df[[date <= split_date for date in salary_df.spp_date]]
    current_ranks = salary_df.groupby('officer_id')['year'].max().to_frame()
    current_ranks = current_ranks.merge(salary_df[['officer_id', 'rank', 'year']],
                                        on=['officer_id', 'year'], how='left')
    officer_df = officer_df.merge(current_ranks, how='left', left_on='id',
                                  right_on='officer_id')
    officer_df = pd.get_dummies(officer_df, columns=['rank'])
    
    return officer_df.drop(columns=['year'])


def gen_allegation_features(officer_df, allegation_df, split_date):
    '''
    '''
    
    allegation_df = allegation_df[allegation_df.incident_date < split_date]
    officer_df['number_complaints'] = 0
    officer_df['pct_officer_complaints'] = 0.0
    officer_df['pct_sustained_complaints'] = 0.0
    off_complaints = allegation_df[allegation_df.is_officer_complaint == True].officer_id.value_counts()  
    sustained = allegation_df[allegation_df.final_finding == 'SU'].officer_id.value_counts()
    count = allegation_df.officer_id.value_counts()
    
    
    for oid in allegation_df.officer_id.unique():
        officer_df.loc[oid, 'number_complaints'] = count[oid]
        
        if oid in off_complaints.index:
            officer_df.loc[oid, 'pct_officer_complaints'] = off_complaints[oid] / count[oid]
        
        if oid in sustained.index:
            officer_df.loc[oid, 'pct_sustained_complaints'] = sustained[oid] / count[oid]

    #internal_allegation_percentile
    
    return officer_df
    

def create_coaccusals_network(officer_alleg_df, allegation_id, officer_id):
    '''
    '''
    G = nx.Graph()
    
    allegations = df[allegation_id]
    
    for aid in allegations.unique():
        officers = df[df.allegation_id == aid]  
        
        if len(officers) > 1:
            oids = officers.officer_id
            
            for oid in oids:
                for oid_2 in oids:
                    if oid != oid_2:
                        if (oid, oid_2) in G.edges():
                            G.edges[oid, oid_2]['count'] += 1
                            G.edges[oid, oid_2]['weight'] = 1 / G.edges[oid, oid_2]['count']

                        else: 
                            G.add_edge(oid, oid_2, count=1, weight=1)
    return G