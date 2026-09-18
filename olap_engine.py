"""
olap_engine.py
================
Generic OLAP cube engine for the donor data warehouse. Supports the four
classic OLAP operations from one query builder (see run_olap_cube), plus a
lightweight get_kpis() for headline numbers that stay visible regardless of
which row/column dimension or measure the person is currently exploring.

  - SLICE      : filters={dim: [single_value]}
  - DICE       : filters={dim1: [v1, v2], dim2: [v3]}
  - ROLL-UP    : caller requests a HIGHER level in the same hierarchy
  - DRILL-DOWN : caller requests a LOWER level in the same hierarchy
  - PIVOT      : caller swaps row_dim <-> col_dim between calls

SCHEMA ASSUMPTION (matches database.py/etl.py as confirmed):
  - dim_donor(donor_id, age, sex, country_code, region, blood_type, bmi, ...)
  - fact_donations(donor_id, eligibility_id, preferred_site,
      last_donation_date, as_of_date, donation_count_last_12m,
      is_regular_donor, lifetime_donation_count, recency_days,
      donation_propensity_score, donated_next_6m, ...)
  - dim_eligibility(eligibility_id, eligibility_status, eligible_to_donate, ...)

NOTE: the geography hierarchy is Region -> Country only. There is no city
column anywhere in dim_donor/etl.py, so a Region -> Country -> City level
isn't backed by real data. If a city column is added to the warehouse later,
add one entry to DIMENSIONS and append "city" to HIERARCHIES["geography"] -
nothing else needs to change.
"""

import pandas as pd

from ml_engine import get_connection

# ==========================================
# DIMENSION REGISTRY
# ==========================================
DIMENSIONS = {
    "region": {
        "label": "Region", "column": "d.region", "hierarchy": "geography", "level": 1,
    },
    "country_code": {
        "label": "Country", "column": "d.country_code", "hierarchy": "geography", "level": 2,
    },
    "blood_type": {
        "label": "Blood Type", "column": "d.blood_type", "hierarchy": None, "level": 1,
    },
    "sex": {
        "label": "Sex", "column": "d.sex", "hierarchy": None, "level": 1,
    },
    "preferred_site": {
        "label": "Donation Site", "column": "f.preferred_site", "hierarchy": None, "level": 1,
    },
    "eligibility_status": {
        "label": "Eligibility Status", "column": "e.eligibility_status", "hierarchy": None, "level": 1,
    },
    "year": {
        "label": "Year",
        "column": "strftime('%Y', f.last_donation_date)",
        "hierarchy": "time", "level": 1,
    },
    "quarter": {
        "label": "Quarter",
        "column": (
            "(strftime('%Y', f.last_donation_date) || '-Q' || "
            "((CAST(strftime('%m', f.last_donation_date) AS INTEGER) + 2) / 3))"
        ),
        "hierarchy": "time", "level": 2,
    },
    "month": {
        "label": "Month",
        "column": "strftime('%Y-%m', f.last_donation_date)",
        "hierarchy": "time", "level": 3,
    },
}

HIERARCHIES = {
    "geography": ["region", "country_code"],
    "time": ["year", "quarter", "month"],
}

# ==========================================
# MEASURE REGISTRY
# ==========================================
MEASURES = {
    "donor_count": {"label": "Donor Count", "sql": "COUNT(DISTINCT f.donor_id)"},
    "avg_recency_days": {"label": "Avg Recency (Days)", "sql": "ROUND(AVG(f.recency_days), 1)"},
    "avg_lifetime_donations": {"label": "Avg Lifetime Donations", "sql": "ROUND(AVG(f.lifetime_donation_count), 1)"},
    "regular_donor_pct": {"label": "Regular Donor %", "sql": "ROUND(AVG(f.is_regular_donor) * 100, 1)"},
    "avg_propensity_score": {"label": "Avg Propensity Score", "sql": "ROUND(AVG(f.donation_propensity_score), 1)"},
    "avg_bmi": {"label": "Avg BMI", "sql": "ROUND(AVG(d.bmi), 1)"},
    "donated_next_6m_pct": {"label": "Predicted Return % (Next 6M)", "sql": "ROUND(AVG(f.donated_next_6m) * 100, 1)"},
}

DEFAULT_MEASURES = ["donor_count", "avg_recency_days", "avg_lifetime_donations", "regular_donor_pct"]

# Fixed set for the KPI cards row - always visible, independent of whatever
# row/col/measure the person is currently exploring in the cube itself.
KPI_MEASURES = ["donor_count", "regular_donor_pct", "avg_recency_days", "donated_next_6m_pct"]

FROM_CLAUSE = """
FROM fact_donations f
JOIN dim_donor d ON f.donor_id = d.donor_id
JOIN dim_eligibility e ON f.eligibility_id = e.eligibility_id
"""


def _validate_dim(key):
    if key not in DIMENSIONS:
        raise ValueError(f"Unknown dimension '{key}'. Valid options: {list(DIMENSIONS)}")
    return key


def _validate_measures(measures):
    ms = measures or DEFAULT_MEASURES
    invalid = [m for m in ms if m not in MEASURES]
    if invalid:
        raise ValueError(f"Unknown measure(s) {invalid}. Valid options: {list(MEASURES)}")
    return ms


def _build_where(filters):
    """Shared by run_olap_cube's main/total queries and get_kpis, so the
    slice/dice filter logic (and its SQL-injection guard) lives in one place."""
    where_clauses = []
    params = []
    for dim_key, values in (filters or {}).items():
        _validate_dim(dim_key)
        if not values:
            continue
        col = DIMENSIONS[dim_key]["column"]
        placeholders = ", ".join(["?"] * len(values))
        where_clauses.append(f"{col} IN ({placeholders})")
        params.extend(values)
    where_stmt = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    return where_stmt, params


def _hierarchy_nav(dim_key):
    """Returns parent/child dim for roll-up/drill-down UI, or Nones for a flat dim."""
    h = DIMENSIONS[dim_key]["hierarchy"]
    if not h:
        return {"name": None, "current_level": dim_key, "parent_dim": None, "child_dim": None}
    levels = HIERARCHIES[h]
    idx = levels.index(dim_key)
    return {
        "name": h,
        "current_level": dim_key,
        "parent_dim": levels[idx - 1] if idx > 0 else None,
        "child_dim": levels[idx + 1] if idx < len(levels) - 1 else None,
    }


def get_meta():
    """Dimension/measure/hierarchy metadata for the frontend to build controls from."""
    return {
        "dimensions": {
            k: {"label": v["label"], "hierarchy": v["hierarchy"], "level": v["level"]}
            for k, v in DIMENSIONS.items()
        },
        "hierarchies": HIERARCHIES,
        "measures": {k: v["label"] for k, v in MEASURES.items()},
        "default_measures": DEFAULT_MEASURES,
        "kpi_measures": KPI_MEASURES,
    }


def get_dimension_values(dimension):
    """Distinct values for a dimension, used to populate slice/dice filter pickers."""
    _validate_dim(dimension)
    col = DIMENSIONS[dimension]["column"]
    conn = get_connection()
    query = f"SELECT DISTINCT {col} AS value {FROM_CLAUSE} WHERE {col} IS NOT NULL ORDER BY value"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df["value"].tolist()


def get_kpis(filters=None):
    """
    Grand-total headline numbers for the KPI cards row: total donors, regular
    donor %, avg recency, and return rate. Responds to the active slice/dice
    filters (so the cards update if the person filters down to a region, for
    example) but is otherwise independent of the row/col dimensions and
    measure the person is exploring in the cube below it.
    """
    ms = _validate_measures(KPI_MEASURES)
    where_stmt, params = _build_where(filters)
    select = ", ".join(f"{MEASURES[m]['sql']} AS {m}" for m in ms)
    query = f"SELECT {select} {FROM_CLAUSE} {where_stmt}"

    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    result = df.to_dict(orient="records")[0] if not df.empty else {m: None for m in ms}
    result["filters"] = filters or {}
    return result


def run_olap_cube(row_dim, col_dim=None, filters=None, measures=None):
    """
    Builds and executes the cube query for the given row/column dimensions,
    slice/dice filters, and measures. Returns records plus a grand-total row
    and hierarchy navigation info (so the frontend knows whether roll-up /
    drill-down is currently possible).
    """
    _validate_dim(row_dim)
    if col_dim:
        _validate_dim(col_dim)
    ms = _validate_measures(measures)
    where_stmt, params = _build_where(filters)

    select_parts = [f"{DIMENSIONS[row_dim]['column']} AS row_value"]
    group_parts = [DIMENSIONS[row_dim]["column"]]
    if col_dim:
        select_parts.append(f"{DIMENSIONS[col_dim]['column']} AS col_value")
        group_parts.append(DIMENSIONS[col_dim]["column"])
    for m in ms:
        select_parts.append(f"{MEASURES[m]['sql']} AS {m}")

    order_stmt = "ORDER BY " + ", ".join(["row_value"] + (["col_value"] if col_dim else []))

    query = f"""
    SELECT {', '.join(select_parts)}
    {FROM_CLAUSE}
    {where_stmt}
    GROUP BY {', '.join(group_parts)}
    {order_stmt}
    """

    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)

    # Grand total, computed independently rather than summed from the group
    # rows, so ratio measures like regular_donor_pct stay a true
    # percentage-of-total rather than an average-of-averages.
    total_select = ", ".join(f"{MEASURES[m]['sql']} AS {m}" for m in ms)
    total_query = f"SELECT {total_select} {FROM_CLAUSE} {where_stmt}"
    totals_df = pd.read_sql_query(total_query, conn, params=params)
    conn.close()

    return {
        "row_dimension": row_dim,
        "col_dimension": col_dim,
        "filters": filters or {},
        "measures": ms,
        "records": df.to_dict(orient="records"),
        "totals": totals_df.to_dict(orient="records")[0] if not totals_df.empty else {},
        "row_hierarchy": _hierarchy_nav(row_dim),
        "col_hierarchy": _hierarchy_nav(col_dim) if col_dim else None,
    }