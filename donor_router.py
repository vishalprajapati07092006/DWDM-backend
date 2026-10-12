"""
donor_router.py
===============
FastAPI router for adding new donors to the warehouse.

Wired into main.py as:
    from donor_router import router as donor_router
    app.include_router(donor_router)

Endpoints:
  POST /api/donors
      -> inserts one donor into dim_donor, dim_eligibility (reused or created),
         and fact_donations. Returns the created donor_id.
"""

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from database import SessionLocal, DimDonor, DimEligibility, FactDonations

router = APIRouter(prefix="/api/donors", tags=["donors"])


# ---------------- Pydantic schema matching actual DB columns ----------------
class DonorCreate(BaseModel):
    # dim_donor
    donor_id: int = Field(..., description="Primary key, must be unique")
    age: int = Field(..., ge=18, le=100)
    sex: str = Field(..., examples=["M", "F"])
    country_code: str = Field(..., examples=["IN", "US"])
    region: str = Field(..., examples=["Asia", "Americas", "Europe"])
    blood_type: str = Field(..., examples=["A+", "O-"])
    is_rare_type: int = Field(0, ge=0, le=1)
    smoker: int = Field(0, ge=0, le=1)
    bmi: float = Field(..., gt=0)
    chronic_condition_flag: int = Field(0, ge=0, le=1)

    # dim_eligibility
    eligible_to_donate: int = Field(1, ge=0, le=1)
    deferral_reason: Optional[str] = None
    eligibility_status: str = Field("eligible", examples=["eligible", "temporary_deferral", "permanent_deferral"])

    # fact_donations
    preferred_site: str = Field(..., examples=["hospital", "mobile_unit", "community_camp"])
    donation_count_last_12m: int = Field(0, ge=0)
    is_regular_donor: int = Field(0, ge=0, le=1)
    years_since_first_donation: int = Field(0, ge=0)
    lifetime_donation_count: int = Field(1, ge=1)
    first_donation_year: int = Field(..., ge=1900, le=2100)
    last_donation_date: str = Field(..., examples=["2024-05-15"])
    recency_days: int = Field(..., ge=0)
    donor_age_at_first_donation: int = Field(..., ge=0)
    blood_type_country_prevalence: float = Field(0.0, ge=0.0, le=1.0)
    donation_propensity_score: float = Field(0.0, ge=0.0)
    donated_next_6m: int = Field(0, ge=0, le=1)
    next_6m_donation_count: int = Field(0, ge=0)


class DonorCreateResponse(BaseModel):
    status: str
    donor_id: int
    eligibility_id: int
    message: str


# ---------------- Endpoint ----------------
@router.post("", response_model=DonorCreateResponse, status_code=201)
def add_donor(payload: DonorCreate):
    db = SessionLocal()
    try:
        # 1. Duplicate check
        existing = db.query(DimDonor).filter(DimDonor.donor_id == payload.donor_id).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Donor ID {payload.donor_id} already exists in the warehouse."
            )

        # 2. Reuse or create eligibility row (dedup by status + eligible flag)
        elig = (
            db.query(DimEligibility)
            .filter(
                DimEligibility.eligibility_status == payload.eligibility_status,
                DimEligibility.eligible_to_donate == payload.eligible_to_donate,
            )
            .first()
        )
        if not elig:
            elig = DimEligibility(
                eligible_to_donate=payload.eligible_to_donate,
                deferral_reason=payload.deferral_reason,
                eligibility_status=payload.eligibility_status,
            )
            db.add(elig)
            db.flush()  # get eligibility_id

        # 3. Insert dim_donor row
        donor = DimDonor(
            donor_id=payload.donor_id,
            age=payload.age,
            sex=payload.sex,
            country_code=payload.country_code,
            region=payload.region,
            blood_type=payload.blood_type,
            is_rare_type=payload.is_rare_type,
            smoker=payload.smoker,
            bmi=payload.bmi,
            chronic_condition_flag=payload.chronic_condition_flag,
        )
        db.add(donor)

        # 4. Insert fact_donations row
        fact = FactDonations(
            donor_id=payload.donor_id,
            eligibility_id=elig.eligibility_id,
            preferred_site=payload.preferred_site,
            donation_count_last_12m=payload.donation_count_last_12m,
            is_regular_donor=payload.is_regular_donor,
            years_since_first_donation=payload.years_since_first_donation,
            lifetime_donation_count=payload.lifetime_donation_count,
            first_donation_year=payload.first_donation_year,
            last_donation_date=payload.last_donation_date,
            recency_days=payload.recency_days,
            donor_age_at_first_donation=payload.donor_age_at_first_donation,
            blood_type_country_prevalence=payload.blood_type_country_prevalence,
            donation_propensity_score=payload.donation_propensity_score,
            as_of_date=str(date.today()),
            donated_next_6m=payload.donated_next_6m,
            next_6m_donation_count=payload.next_6m_donation_count,
        )
        db.add(fact)

        db.commit()
        return DonorCreateResponse(
            status="success",
            donor_id=payload.donor_id,
            eligibility_id=elig.eligibility_id,
            message=f"Donor {payload.donor_id} added to dim_donor + fact_donations.",
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected insert error: {str(e)}")
    finally:
        db.close()