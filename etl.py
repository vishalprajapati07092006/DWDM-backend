import pandas as pd
import sqlite3
import os

def run_etl(csv_path="data/blood_donation_registry_ml_ready (1).csv", db_path="data_warehouse.db"):
    # Fallback path check
    if not os.path.exists(csv_path):
        if os.path.exists("blood_donation_registry_ml_ready (1).csv"):
            csv_path = "blood_donation_registry_ml_ready (1).csv"
        else:
            print(f"❌ Error: Could not find dataset at '{csv_path}'")
            return

    print("1/4 Loading raw blood donation registry dataset...")
    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)

    print("2/4 Populating Dim_Donor...")
    donor_cols = ['donor_id', 'age', 'sex', 'country_code', 'region', 'blood_type', 
                  'is_rare_type', 'smoker', 'bmi', 'chronic_condition_flag']
    donors = df[donor_cols].drop_duplicates(subset=['donor_id'])
    donors.to_sql('dim_donor', conn, if_exists='replace', index=False)

    print("3/4 Populating Dim_Eligibility...")
    eligibility_cols = ['eligible_to_donate', 'deferral_reason', 'eligibility_status']
    eligibility = df[eligibility_cols].drop_duplicates().reset_index(drop=True)
    eligibility['eligibility_id'] = eligibility.index + 1
    eligibility.to_sql('dim_eligibility', conn, if_exists='replace', index=False)

    print("4/4 Building and populating Fact_Donations...")
    df = df.merge(eligibility, on=eligibility_cols, how='left')

    fact_cols = [
        'donor_id', 'eligibility_id', 'preferred_site', 'donation_count_last_12m',
        'is_regular_donor', 'years_since_first_donation', 'lifetime_donation_count',
        'first_donation_year', 'last_donation_date', 'recency_days',
        'donor_age_at_first_donation', 'blood_type_country_prevalence',
        'donation_propensity_score', 'as_of_date', 'donated_next_6m',
        'next_6m_donation_count'
    ]

    fact_df = df[fact_cols]
    fact_df.to_sql('fact_donations', conn, if_exists='replace', index=False)

    conn.commit()
    conn.close()
    print("✅ ETL Complete! Data Warehouse populated with Blood Donation records.")

if __name__ == "__main__":
    run_etl()