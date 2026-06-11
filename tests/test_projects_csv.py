"""Tests for projects_from_csv: happy path, missing columns, empty file, bad numerics."""
from __future__ import annotations

import io
import textwrap

import pytest

from src.projects import Project, REQUIRED_CSV_COLUMNS, projects_from_csv

# ---------------------------------------------------------------------------
# Minimal valid CSV (one row, all required columns present)
# ---------------------------------------------------------------------------
_VALID_ROW = (
    "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
    "opex_per_bbl,nri,pc,rig_days,earliest_quarter\n"
    "P001,Test-001,new_drill,Midland-S,9000000,800,1.4,0.9,12,0.75,0.9,30,1\n"
)


def _csv(content: str):
    """Return a file-like object that pandas.read_csv accepts."""
    return io.StringIO(content)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_csv_returns_projects():
    projects = projects_from_csv(_csv(_VALID_ROW))
    assert len(projects) == 1
    p = projects[0]
    assert isinstance(p, Project)
    assert p.project_id == "P001"
    assert p.category == "new_drill"
    assert p.capex_usd == pytest.approx(9_000_000)
    assert p.earliest_quarter == 1
    assert isinstance(p.earliest_quarter, int)


def test_multi_row_csv():
    content = (
        "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
        "opex_per_bbl,nri,pc,rig_days,earliest_quarter\n"
        "P001,Test-001,new_drill,Midland-S,9000000,800,1.4,0.9,12,0.75,0.9,30,1\n"
        "P002,Test-002,workover,Delaware-N,1500000,120,1.1,0.5,8,0.80,0.7,5,2\n"
    )
    projects = projects_from_csv(_csv(content))
    assert len(projects) == 2
    assert projects[1].project_id == "P002"


def test_real_synthetic_csv_loads():
    """Round-trip the committed synthetic backlog through projects_from_csv."""
    projects = projects_from_csv("data/synthetic/projects.csv")
    assert len(projects) > 0
    for p in projects:
        assert isinstance(p, Project)
        assert p.earliest_quarter in (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------

def test_missing_single_column_raises():
    # Drop 'rig_days' from the header.
    content = (
        "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
        "opex_per_bbl,nri,pc,earliest_quarter\n"
        "P001,Test-001,new_drill,Midland-S,9000000,800,1.4,0.9,12,0.75,0.9,1\n"
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        projects_from_csv(_csv(content))


def test_missing_multiple_columns_lists_them():
    # Keep only project_id and name — every numeric column is absent.
    content = "project_id,name\nP001,Test-001\n"
    with pytest.raises(ValueError) as exc_info:
        projects_from_csv(_csv(content))
    msg = str(exc_info.value)
    assert "Missing required columns" in msg
    assert "capex_usd" in msg


def test_required_columns_constant_matches_dataclass():
    """REQUIRED_CSV_COLUMNS must cover every Project field (no drift)."""
    from dataclasses import fields as dc_fields
    field_names = {f.name for f in dc_fields(Project)}
    assert field_names == set(REQUIRED_CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Empty file
# ---------------------------------------------------------------------------

def test_empty_data_raises():
    # Header only, no data rows.
    content = (
        "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
        "opex_per_bbl,nri,pc,rig_days,earliest_quarter\n"
    )
    with pytest.raises(ValueError, match="no data rows"):
        projects_from_csv(_csv(content))


# ---------------------------------------------------------------------------
# Bad numeric values
# ---------------------------------------------------------------------------

def test_non_numeric_value_raises():
    content = (
        "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
        "opex_per_bbl,nri,pc,rig_days,earliest_quarter\n"
        "P001,Test-001,new_drill,Midland-S,NINE_MILLION,800,1.4,0.9,12,0.75,0.9,30,1\n"
    )
    with pytest.raises(ValueError, match="Non-numeric"):
        projects_from_csv(_csv(content))


def test_blank_numeric_cell_raises():
    content = (
        "project_id,name,category,area,capex_usd,qi_bopd,di_annual,b,"
        "opex_per_bbl,nri,pc,rig_days,earliest_quarter\n"
        "P001,Test-001,new_drill,Midland-S,,800,1.4,0.9,12,0.75,0.9,30,1\n"
    )
    with pytest.raises(ValueError, match="Non-numeric"):
        projects_from_csv(_csv(content))
