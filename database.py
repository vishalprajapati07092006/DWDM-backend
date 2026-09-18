import sqlite3
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./data_warehouse.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DIMENSION TABLES ---
class DimDonor(Base):
    __tablename__ = "dim_donor"
    donor_id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    sex = Column(String)
    country_code = Column(String)
    region = Column(String)
    blood_type = Column(String)
    is_rare_type = Column(Integer)
    smoker = Column(Integer)
    bmi = Column(Float)
    chronic_condition_flag = Column(Integer)

class DimEligibility(Base):
    __tablename__ = "dim_eligibility"
    eligibility_id = Column(Integer, primary_key=True, autoincrement=True)
    eligible_to_donate = Column(Integer)
    deferral_reason = Column(String, nullable=True)
    eligibility_status = Column(String)

# --- FACT TABLE ---
class FactDonations(Base):
    __tablename__ = "fact_donations"
    donor_id = Column(Integer, ForeignKey("dim_donor.donor_id"), primary_key=True)
    eligibility_id = Column(Integer, ForeignKey("dim_eligibility.eligibility_id"))

    # Operations & RFM Measures
    preferred_site = Column(String)
    donation_count_last_12m = Column(Integer)
    is_regular_donor = Column(Integer)
    years_since_first_donation = Column(Integer)
    lifetime_donation_count = Column(Integer)
    first_donation_year = Column(Integer)
    last_donation_date = Column(String)
    recency_days = Column(Integer)
    donor_age_at_first_donation = Column(Integer)
    blood_type_country_prevalence = Column(Float)
    donation_propensity_score = Column(Float)
    as_of_date = Column(String)

    # Classification & ML Prediction Outputs
    donated_next_6m = Column(Integer)
    next_6m_donation_count = Column(Integer)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("✅ Blood Donation Data Warehouse Star Schema created successfully in data_warehouse.db")