"""
Tests for the source-ingestion progress helper.

``_compute_ingestion_progress`` is the pure function that turns four
observable signals (command status, full-text present, chunk count,
insight count) into a (stage, percent) pair the frontend can render.
Keeping this pure lets us test every branch without spinning up
surreal-commands or SurrealDB.
"""

from __future__ import annotations

import pytest

from open_notebook.domain.notebook import (
    INGESTION_STAGES,
    _compute_ingestion_progress,
)


def _call(
    *,
    status=None,
    text=False,
    chunks=0,
    insights=0,
):
    return _compute_ingestion_progress(
        command_status=status,
        has_full_text=text,
        embedded_chunks=chunks,
        insights_count=insights,
    )


def test_completed_is_full():
    assert _call(status="completed", text=True, chunks=10, insights=3) == (
        "completed",
        100,
    )


def test_failed_stage():
    stage, pct = _call(status="failed", text=False)
    assert stage == "failed"
    assert pct == 0


@pytest.mark.parametrize("status", [None, "queued", "new"])
def test_queued_states(status):
    """No command yet, or queued but not running → lowest tier."""
    stage, pct = _call(status=status)
    assert stage == "queued"
    assert pct == 5


def test_running_no_text_is_extracting():
    stage, pct = _call(status="running", text=False)
    assert stage == "extracting"
    assert pct == 25


def test_running_with_text_no_chunks_is_embedding():
    stage, pct = _call(status="running", text=True, chunks=0)
    assert stage == "embedding"
    assert pct == 55


def test_running_with_chunks_no_insights_is_transforming():
    stage, pct = _call(status="running", text=True, chunks=12, insights=0)
    assert stage == "transforming"
    assert pct == 85


def test_running_with_insights_but_not_done_is_near_done():
    """Chunks + insights but command still 'running' → 95%, not 100%."""
    stage, pct = _call(status="running", text=True, chunks=12, insights=2)
    assert stage == "transforming"
    assert pct == 95


def test_unknown_status_falls_through_to_state_inference():
    """Anything not in the explicit list is treated as 'in flight'."""
    stage, pct = _call(status="unknown", text=False)
    assert stage == "extracting"
    assert pct == 25


def test_monotonic_ordering_of_percentages():
    """Every stage a running job passes through is a strict step up."""
    order = [
        _call(status="queued")[1],
        _call(status="running", text=False)[1],
        _call(status="running", text=True, chunks=0)[1],
        _call(status="running", text=True, chunks=5, insights=0)[1],
        _call(status="running", text=True, chunks=5, insights=2)[1],
        _call(status="completed", text=True, chunks=5, insights=2)[1],
    ]
    assert order == sorted(order)
    assert order[0] < order[-1]


def test_all_returned_stages_are_declared():
    """The function should never return an undeclared stage string."""
    seen = {
        _call(status="completed")[0],
        _call(status="failed")[0],
        _call(status="queued")[0],
        _call(status="running", text=False)[0],
        _call(status="running", text=True, chunks=0)[0],
        _call(status="running", text=True, chunks=5, insights=0)[0],
    }
    for s in seen:
        assert s in INGESTION_STAGES
