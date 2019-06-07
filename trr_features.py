
def create_trr_outcome(officer_df, allegation_df, end_date_train):
    outcome_window = allegation_df.loc[
        allegation_df.incident_date > end_date_train]

    officer_df['sustained_outcome'] = officer_df.apply(
        lambda x: 1 if x['id'] in list(sustained.officer_id) else 0, axis=1)
    return officer_df