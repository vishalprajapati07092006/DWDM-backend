import sqlite3
import json
from olap_router import router as olap_router
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from ml_engine import (
    train_classification_model,
    run_kmeans_clustering,
    run_association_rules,
    get_connection,
    FEATURE_COLS,
    DB_PATH,
    DATASET_CAVEAT,
)

ml_registry = {}
PR_CURVE_PATH = "pr_curve.json"

# ============================================================
# CURATED RULE SELECTION - matches the 8 rules chosen in the
# Colab notebook (project1212.ipynb). Each entry maps the human
# antecedent/consequent pattern to the raw item-set labels that
# FP-Growth produces, so mined metrics can be looked up per rule.
# ============================================================
CURATED_RULES = [
    {
        "id": "r1",
        "antecedents": ["recency<6mo", "age<30"],
        "consequents": ["donated_next_6m=Yes"],
        "strategy": "Campus & Youth Recurrent Engagement Drives",
        # Fallback values from the notebook if the mined pattern isn't present
        "fallback": {"support": 0.0002, "confidence": 0.7000, "lift": 1.8861},
    },
    {
        "id": "r2",
        "antecedents": ["lifetime_donations=11+", "recency<6mo"],
        "consequents": ["donated_next_6m=Yes"],
        "strategy": "VIP Fast-Track & Super-Donor Loyalty Rewards",
        "fallback": {"support": 0.0034, "confidence": 0.6561, "lift": 1.7677},
    },
    {
        "id": "r3",
        "antecedents": ["bmi=underweight"],
        "consequents": ["donated_next_6m=No"],
        "strategy": "Nurse Iron & Dietary Counseling Follow-Up",
        "fallback": {"support": 0.0273, "confidence": 0.9403, "lift": 1.4952},
    },
    {
        "id": "r4",
        "antecedents": ["age=60+", "recency=2yr+"],
        "consequents": ["donated_next_6m=No"],
        "strategy": "Senior Care Mobile Outreach & Health Assessment",
        "fallback": {"support": 0.0736, "confidence": 0.7464, "lift": 1.1868},
    },
    {
        "id": "r5",
        "antecedents": ["age=60+", "smoker=Yes"],
        "consequents": ["donated_next_6m=No"],
        "strategy": "Targeted Preventive Health & Gentle Engagement",
        "fallback": {"support": 0.0175, "confidence": 0.7450, "lift": 1.1847},
    },
    {
        "id": "r6",
        "antecedents": ["smoker=Yes", "recency=2yr+"],
        "consequents": ["donated_next_6m=No"],
        "strategy": "Reactivation Campaign with Wellness Incentives",
        "fallback": {"support": 0.1572, "confidence": 0.6710, "lift": 1.0670},
    },
    {
        "id": "r7",
        "antecedents": ["lifetime_donations=11+", "recency=2yr+"],
        "consequents": ["donated_next_6m=No"],
        "strategy": "Hero Donor Win-Back & Personalized Thank-You Drive",
        "fallback": {"support": 0.6217, "confidence": 0.6380, "lift": 1.0145},
    },
    
]


def _build_transactions_from_db():
    """
    Builds the transaction list used by FP-Growth, exactly mirroring
    the item labels the notebook's antecedent/consequent names use.
    This is the mining input - the algorithm runs on it, the curated
    rules then get matched against its output.
    """
    conn = get_connection()
    query = """
    SELECT d.age, d.bmi, d.smoker, d.chronic_condition_flag, d.blood_type,
           e.eligibility_status, f.recency_days, f.lifetime_donation_count,
           f.is_regular_donor, f.donated_next_6m
    FROM fact_donations f
    JOIN dim_donor d ON f.donor_id = d.donor_id
    JOIN dim_eligibility e ON f.eligibility_id = e.eligibility_id
    LIMIT 20000
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    transactions = []
    for _, row in df.iterrows():
        items = []

        # Donor attributes
        if pd.notna(row.get('age')):
            if row['age'] < 30:
                items.append("age<30")
            elif row['age'] >= 60:
                items.append("age=60+")

        if pd.notna(row.get('bmi')) and row['bmi'] < 18.5:
            items.append("bmi=underweight")

        if row.get('smoker') == 1:
            items.append("smoker=Yes")

        if row.get('chronic_condition_flag') == 1:
            items.append("Has_Chronic_Condition")

        items.append(f"BloodType_{row['blood_type']}")
        items.append(f"Status_{row['eligibility_status']}")

        # Donation behavior
        if pd.notna(row.get('recency_days')):
            if row['recency_days'] < 180:
                items.append("recency<6mo")
            if row['recency_days'] >= 730:
                items.append("recency=2yr+")

        if pd.notna(row.get('lifetime_donation_count')) and row['lifetime_donation_count'] >= 11:
            items.append("lifetime_donations=11+")

        if row.get('is_regular_donor') == 1:
            items.append("Regular_Donor")

        # Outcome
        if row.get('donated_next_6m') == 1:
            items.append("donated_next_6m=Yes")
        else:
            items.append("donated_next_6m=No")

        transactions.append(items)

    return transactions


def _mine_and_select_curated_rules(min_support=0.001, min_confidence=0.1):
    """
    Runs FP-Growth on the real warehouse data (like the notebook did),
    then selects the 8 curated rules. For each curated rule, if the
    mined output contains an equivalent itemset, the mined
    support/confidence/lift are used; otherwise the notebook's
    fallback values are used.
    """
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    transactions = _build_transactions_from_db()
    if not transactions:
        return [dict(**{k: v for k, v in r.items() if k != "fallback"},
                     **r["fallback"]) for r in CURATED_RULES]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    # Lower thresholds than the notebook's defaults so all 8 curated
    # patterns (including the rare ones) have a chance to appear in
    # the mined output and get real, freshly-computed metrics.
    frequent_itemsets = fpgrowth(df_tf, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        mined_lookup = {}
    else:
        mined_rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=min_confidence
        )
        # Build a lookup keyed by (frozenset(antecedents), frozenset(consequents))
        # for O(1) matching against the curated rules below.
        mined_lookup = {}
        for _, row in mined_rules.iterrows():
            key = (frozenset(row['antecedents']), frozenset(row['consequents']))
            mined_lookup[key] = {
                "support": round(float(row['support']), 4),
                "confidence": round(float(row['confidence']), 4),
                "lift": round(float(row['lift']), 4),
            }

    selected = []
    for rule in CURATED_RULES:
        ant_key = frozenset(rule["antecedents"])
        cons_key = frozenset(rule["consequents"])
        mined = mined_lookup.get((ant_key, cons_key))

        metrics = mined if mined else rule["fallback"]
        selected.append({
            "antecedents": rule["antecedents"],
            "consequents": rule["consequents"],
            "support": metrics["support"],
            "confidence": metrics["confidence"],
            "lift": metrics["lift"],
            "strategy": rule["strategy"],
            "source": "mined" if mined else "notebook_fallback",
        })

    # Sort by Lift descending, same ordering as the notebook output table
    selected.sort(key=lambda r: r["lift"], reverse=True)
    return selected

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Initializing Analytics Engine: Running Programmatic Threshold Optimization...")
    try:
        # Dynamically finds the optimal sweet-spot threshold on server startup,
        # and now also persists the full precision/recall/threshold curve to
        # PR_CURVE_PATH for the /api/mining/pr-curve endpoint below.
        clf_artifacts = train_classification_model(beta=1.5, save_pr_curve_path=PR_CURVE_PATH)
        ml_registry["clf_model"] = clf_artifacts["model"]
        ml_registry["clf_metrics"] = clf_artifacts["metrics"]
        ml_registry["feature_importances"] = clf_artifacts["feature_importances"]
        ml_registry["threshold"] = clf_artifacts["threshold"]
        ml_registry["dataset_caveat"] = clf_artifacts["dataset_caveat"]

        ml_registry["association_rules"] = run_association_rules()
        print(f"✅ Analytics Engine Ready (Optimal Operating Threshold = {ml_registry['threshold']})")
    except Exception as e:
        print(f"❌ Startup Error: {str(e)}")
        raise e

    yield
    ml_registry.clear()

app = FastAPI(title="Blood Donation DSS Analytics Engine", lifespan=lifespan)
app.include_router(olap_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    age: int = Field(..., examples=[45])
    bmi: float = Field(..., examples=[26.0])
    donation_count_last_12m: int = Field(..., examples=[2])
    years_since_first_donation: int = Field(..., examples=[5])
    lifetime_donation_count: int = Field(..., examples=[12])
    recency_days: int = Field(..., examples=[60])
    is_regular_donor: int = Field(..., examples=[1])
    eligible_to_donate: int = Field(..., examples=[1])

class PredictResponse(BaseModel):
    risk_probability: float
    will_return: bool
    status: str

# NEW: mirrors PredictRequest, but scoped to CLUSTER_FEATURE_COLS - the six
# fields the KMeans model was actually fit on. n_clusters is included so the
# simulator always scores the new donor against the same k the dashboard is
# currently displaying (rather than a hardcoded k that could disagree with
# what's on screen).
class ClusterPredictRequest(BaseModel):
    age: float = Field(..., examples=[35])
    bmi: float = Field(..., examples=[24.0])
    recency_days: float = Field(..., examples=[90])
    donation_count_last_12m: float = Field(..., examples=[2])
    lifetime_donation_count: float = Field(..., examples=[8])
    donation_propensity_score: float = Field(..., examples=[45])
    n_clusters: int = Field(4, ge=2, le=6, examples=[4])

class ClusterPredictResponse(BaseModel):
    assigned_cluster: int
    cluster_profile: dict
    distances_to_centroids: List[float]

@app.get("/api/olap/pivot")
def get_olap_pivot(region: Optional[str] = "All", blood_type: Optional[str] = "All"):
    try:
        conn = get_connection()
        where_clauses = []
        params = []

        if region != "All":
            where_clauses.append("d.region = ?")
            params.append(region)
        if blood_type != "All":
            where_clauses.append("d.blood_type = ?")
            params.append(blood_type)

        where_stmt = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        query = f"""
        SELECT 
            d.region,
            d.blood_type,
            COUNT(DISTINCT f.donor_id) AS donor_count,
            ROUND(AVG(f.recency_days), 1) AS avg_recency_days,
            ROUND(AVG(f.lifetime_donation_count), 1) AS avg_lifetime_donations,
            ROUND(AVG(f.is_regular_donor) * 100, 1) AS regular_donor_percentage
        FROM fact_donations f
        JOIN dim_donor d ON f.donor_id = d.donor_id
        {where_stmt}
        GROUP BY d.region, d.blood_type
        ORDER BY d.region, d.blood_type
        """

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OLAP Query Error: {str(e)}")

@app.get("/api/mining/classification")
def get_classification_metrics():
    if "clf_metrics" not in ml_registry:
        raise HTTPException(status_code=503, detail="Model uninitialized.")

    return {
        "accuracy": ml_registry["clf_metrics"]["accuracy"],
        "recall": ml_registry["clf_metrics"]["recall"],
        "metrics": ml_registry["clf_metrics"],
        "feature_importances": ml_registry["feature_importances"],
        "threshold": ml_registry["threshold"],
        # Surfaced so the frontend/report can show this next to the numbers
        # rather than presenting them as an unqualified real-world guarantee.
        "dataset_caveat": ml_registry["dataset_caveat"],
    }

@app.get("/api/mining/pr-curve")
def get_pr_curve():
    """
    Returns the full precision/recall-vs-threshold curve computed during
    training, plus the chosen operating point, so the frontend can render
    the curve and mark where the beta=1.5 threshold sits on it.
    """
    try:
        with open(PR_CURVE_PATH, "r") as f:
            curve = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="PR curve not yet computed.")

    # Downsample for a lighter payload/chart - the raw curve can have
    # thousands of points, which is more than any chart needs to render.
    n = len(curve["thresholds"])
    if n > 60:
        idx = np.linspace(0, n - 1, 60).astype(int)
        curve["thresholds"] = [curve["thresholds"][i] for i in idx]
        curve["precisions"] = [curve["precisions"][i] for i in idx]
        curve["recalls"] = [curve["recalls"][i] for i in idx]

    return curve

@app.get("/api/mining/clustering")
def get_clustering_results(clusters: int = Query(4, ge=2, le=6)):
    """
    NOTE: response shape changed. run_kmeans_clustering now returns a dict
    with "clusters" (the per-cluster profile, same fields as before plus
    size/pct_of_total), "multicollinearity_check" (full pairwise correlation
    check across CLUSTER_FEATURE_COLS, not just the one pair already known
    from EDA), and "caveat" (the donation_propensity_score disclosure).
    Frontend code reading this as a bare array needs to read `.clusters`
    instead - see donor_retention_App.jsx's fetchMiningData.
    """
    try:
        return run_kmeans_clustering(n_clusters=clusters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering Error: {str(e)}")

@app.post("/api/mining/cluster-predict", response_model=ClusterPredictResponse)
def predict_cluster_membership(payload: ClusterPredictRequest):
    """
    Live "which segment does this donor fall into" inference - the
    clustering counterpart to /api/mining/predict. Refits KMeans on the
    current warehouse data with the requested k (same deterministic
    random_state=42 used everywhere else in this module, so results are
    stable and match whatever the dashboard is currently displaying for that
    k), then scores the submitted donor against it.
    """
    try:
        payload_dict = payload.dict()
        n_clusters = payload_dict.pop("n_clusters")
        result = run_kmeans_clustering(n_clusters=n_clusters, new_point=payload_dict)
        prediction = result.get("prediction")
        if not prediction:
            raise HTTPException(status_code=500, detail="Cluster prediction was not computed.")
        return prediction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cluster Prediction Error: {str(e)}")

@app.get("/api/mining/association")
def get_association_rules_endpoint():
    """
    Runs FP-Growth on the live warehouse (same algorithm as the
    notebook), then returns only the 8 curated multi-concept rules
    chosen in the notebook. Each rule carries a `source` field:
    'mined' when the pattern was found in the algorithm's output
    (metrics freshly computed from the DB), 'notebook_fallback'
    when the pattern wasn't present and the notebook's pre-computed
    values are used instead.
    """
    try:
        return _mine_and_select_curated_rules()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Association Rule Mining Error: {str(e)}")

@app.post("/api/mining/predict", response_model=PredictResponse)
def predict_donor_return(payload: PredictRequest):
    clf = ml_registry.get("clf_model")
    threshold = ml_registry.get("threshold", 0.5)

    if clf is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    try:
        input_data = pd.DataFrame([{
            'age': payload.age,
            'bmi': payload.bmi,
            'donation_count_last_12m': payload.donation_count_last_12m,
            'years_since_first_donation': payload.years_since_first_donation,
            'lifetime_donation_count': payload.lifetime_donation_count,
            'recency_days': payload.recency_days,
            'is_regular_donor': payload.is_regular_donor,
            'eligible_to_donate': payload.eligible_to_donate
        }])[FEATURE_COLS]

        probabilities = clf.predict_proba(input_data)
        return_prob = float(probabilities[0][1])

        # Inference uses calculated optimal threshold
        will_return = return_prob >= threshold

        return PredictResponse(
            risk_probability=round(return_prob, 4),
            will_return=will_return,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")