"""
Tests for the smart model router in ``open_notebook.graphs.chat._route_model``.

The router is a pure function guarded by two env vars
(``OPEN_NOTEBOOK_ROUTER_CHEAP_MODEL`` / ``OPEN_NOTEBOOK_ROUTER_STRONG_MODEL``).
At import time those are empty in CI, so the module-level ``_ROUTER_ENABLED``
flag is False and the router no-ops. These tests patch the module-level
constants directly to exercise both the enabled and disabled paths without
needing to reload the module.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from open_notebook.graphs import chat as chat_module


@pytest.fixture(autouse=True)
def _router_enabled():
    """Force the router on with sentinel model IDs for the test run."""
    with (
        patch.object(chat_module, "_ROUTER_ENABLED", True),
        patch.object(chat_module, "_ROUTER_CHEAP_MODEL", "cheap:model"),
        patch.object(chat_module, "_ROUTER_STRONG_MODEL", "strong:model"),
    ):
        yield


def test_short_factual_question_routes_to_cheap():
    """'What year was X founded?' — short, no reasoning keywords → cheap."""
    chosen = chat_module._route_model(
        question="What year was GitHub founded?",
        context_chars=1500,
        excerpt_count=1,
    )
    assert chosen == "cheap:model"


def test_reasoning_keyword_routes_to_strong():
    """'Explain why …' triggers the reasoning pattern → strong."""
    chosen = chat_module._route_model(
        question="Explain why transformers scale better than RNNs.",
        context_chars=1500,
        excerpt_count=1,
    )
    assert chosen == "strong:model"


def test_long_question_routes_to_strong():
    """A verbose question above the length threshold → strong."""
    long_q = (
        "I'm trying to understand the relationship between three different "
        "topics I've been reading about this week — specifically the way "
        "that retrieval augmented generation, long-context models, and "
        "agentic tool use all attempt to solve slightly different failure "
        "modes of plain LLM prompting. Can you walk me through it?"
    )
    assert len(long_q) >= chat_module.ROUTER_LONG_QUESTION_CHARS
    chosen = chat_module._route_model(
        question=long_q,
        context_chars=1500,
        excerpt_count=1,
    )
    assert chosen == "strong:model"


def test_many_excerpts_routes_to_strong():
    """Multi-document synthesis — 4+ excerpts → strong even for short qs."""
    chosen = chat_module._route_model(
        question="Summarize.",
        context_chars=1500,
        excerpt_count=chat_module.ROUTER_MULTI_SOURCE_THRESHOLD,
    )
    assert chosen == "strong:model"


def test_large_context_routes_to_strong():
    """Huge system prompt → probably big sources selected → strong."""
    chosen = chat_module._route_model(
        question="What's the main point?",
        context_chars=chat_module.ROUTER_LARGE_CONTEXT_CHARS,
        excerpt_count=1,
    )
    assert chosen == "strong:model"


def test_word_boundary_prevents_false_match():
    """'explainer' must NOT trigger on 'explain' (word-boundary regex)."""
    chosen = chat_module._route_model(
        question="Does the docs explainer look OK?",
        context_chars=500,
        excerpt_count=1,
    )
    # 'explainer' is one word — boundary regex should treat it as non-match.
    # But we still expect cheap because no *other* signal fires.
    assert chosen == "cheap:model"


def test_router_disabled_returns_none():
    """No env vars → router opts out so the caller keeps its choice."""
    with (
        patch.object(chat_module, "_ROUTER_ENABLED", False),
        patch.object(chat_module, "_ROUTER_CHEAP_MODEL", None),
        patch.object(chat_module, "_ROUTER_STRONG_MODEL", None),
    ):
        chosen = chat_module._route_model(
            question="anything",
            context_chars=0,
            excerpt_count=0,
        )
    assert chosen is None


def test_empty_question_routes_to_cheap():
    """Nothing to analyze → don't pay for the flagship."""
    chosen = chat_module._route_model(
        question="   ", context_chars=0, excerpt_count=0
    )
    assert chosen == "cheap:model"
