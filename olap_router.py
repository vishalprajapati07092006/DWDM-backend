"""
olap.py
========
FastAPI router wrapping olap_engine.py.

Wired into main.py as:
    from olap import router as olap_router
    app.include_router(olap_router)

Endpoints:
  GET /api/olap/meta
      -> dimension/measure/hierarchy metadata for building the UI controls.

  GET /api/olap/kpis?filters={...}
      -> fixed headline numbers (total donors, regular donor %, avg recency,
         return rate) for the KPI cards row. Respects slice/dice filters but
         is independent of whatever row/col dimension is currently selected.

  GET /api/olap/dimension-values?dimension=blood_type
      -> distinct values for one dimension, to populate a slice/dice filter
         picker.

  GET /api/olap/cube?row_dim=region&col_dim=blood_type&filters={...}&measures=donor_count
      -> the cube query itself. row_dim is required; col_dim, filters and
         measures are optional.
      - filters is a JSON-encoded object, e.g. {"region": ["Americas"], "blood_type": ["A+","O+"]}
      - measures is a comma-separated list of measure keys
"""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from olap_engine import get_meta, get_dimension_values, run_olap_cube, get_kpis

router = APIRouter(prefix="/api/olap", tags=["olap"])


def _parse_filters(filters: Optional[str]):
    try:
        return json.loads(filters) if filters else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="filters must be valid JSON")


@router.get("/meta")
def olap_meta():
    return get_meta()


@router.get("/kpis")
def olap_kpis(filters: Optional[str] = Query(None, description='JSON-encoded {"dimension": [values]}')):
    parsed_filters = _parse_filters(filters)
    try:
        return get_kpis(parsed_filters)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KPI Error: {str(e)}")


@router.get("/dimension-values")
def olap_dimension_values(dimension: str):
    try:
        return {"dimension": dimension, "values": get_dimension_values(dimension)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dimension Values Error: {str(e)}")


@router.get("/cube")
def olap_cube(
    row_dim: str = Query(..., description="Dimension for rows, e.g. 'region'"),
    col_dim: Optional[str] = Query(None, description="Optional dimension for columns (cross-tab/pivot)"),
    filters: Optional[str] = Query(None, description='JSON-encoded {"dimension": [values]} for slice/dice'),
    measures: Optional[str] = Query(None, description="Comma-separated measure keys"),
):
    parsed_filters = _parse_filters(filters)
    parsed_measures = [m.strip() for m in measures.split(",")] if measures else None

    try:
        return run_olap_cube(row_dim, col_dim, parsed_filters, parsed_measures)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OLAP Cube Error: {str(e)}")