"""
Tests for the LLM-verified semantic cache tier in ``open_notebook.domain.chat_cache``.

We focus on the *new* verification branch (tier 3) because tiers 1 and 2 are
covered implicitly by integration with the chat router. The verifier lives
at module level so tests can monkeypatch it without touching the DB layer.

The tests mock:
  * ``repo_query`` — to simulate candidate rows being present and the
    semantic query returning a row with a controlled ``similarity``.
  * ``generate_embedding`` — so we don't need an embedding provider online.
  * ``_verify_cache_match_with_llm`` — so we can assert it's invoked only
    in the danger band and that its verdict gates the cache hit.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from open_notebook.domain import chat_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo_query_stub(candidates: int, similarity: float):
    """Return an async function that mimics ``repo_query`` for our two calls.

    Call 1 (exact-match tier 1): returns [] (simulate miss so we exercise
      tier 2).
    Call 2 (probe count): returns [{"n": candidates}] — controls whether
      the semantic tier is even attempted.
    Call 3 (semantic select): returns a single row with the supplied
      similarity + enough fields for ``_normalize_row`` to succeed.
    """
    calls = {"i": 0}

    async def stub(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        calls["i"] += 1
        q = query.strip()
        if q.startswith("SELECT * FROM chat_qa_cache") and "question_norm" in q:
            return []  # tier 1 miss
        if "count()" in q:
            return [{"n": candidates}]
        if "vector::similarity::cosine" in q:
            return [
                {
                    "id": "chat_qa_cache:abc",
                    "question": "How do I use feature X on Linux?",
                    "answer": "Cached answer body.",
                    "similarity": similarity,
                    "hit_count": 0,
                    "notebook_id": "notebook:nb1",
                    "context_fingerprint": "fp",
                }
            ]
        return []

    return stub, calls


# ---------------------------------------------------------------------------
# Verifier parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verifier_returns_true_only_on_yes():
    """A clean 'YES' verdict → trust the cache; anything else → reject.

    Explicitly covers the prefix parser: 'YES.', 'yes ' and newline variants
    should all pass; 'NO, but …' and garbage must not."""

    async def fake_invoke_yes(_p):
        class R:
            content = "YES"

        return R()

    async def fake_invoke_yes_punct(_p):
        class R:
            content = "  Yes.\n"

        return R()

    async def fake_invoke_no(_p):
        class R:
            content = "NO, the question is about macOS."

        return R()

    async def fake_invoke_noise(_p):
        class R:
            content = "Maybe, but actually it depends…"

        return R()

    # Patch provision_langchain_model to return a dummy model object whose
    # ainvoke is swapped in per test case.
    class DummyModel:
        def __init__(self, fn):
            self.ainvoke = fn

    cases = [
        (fake_invoke_yes, True),
        (fake_invoke_yes_punct, True),
        (fake_invoke_no, False),
        (fake_invoke_noise, False),
    ]
    for fn, expected in cases:
        with patch(
            "open_notebook.ai.provision.provision_langchain_model",
            new=AsyncMock(return_value=DummyModel(fn)),
        ):
            got = await chat_cache._verify_cache_match_with_llm(
                new_question="How do I use feature X on macOS?",
                cached_question="How do I use feature X on Linux?",
                cached_answer="Cached answer body.",
            )
        assert got is expected, f"expected {expected} for fn {fn.__name__}"


@pytest.mark.asyncio
async def test_verifier_timeout_returns_false():
    """A slow LLM should degrade to 'cache miss', not block the request."""
    import asyncio

    class SlowModel:
        async def ainvoke(self, _p):
            await asyncio.sleep(5.0)

    with patch(
        "open_notebook.ai.provision.provision_langchain_model",
        new=AsyncMock(return_value=SlowModel()),
    ):
        got = await chat_cache._verify_cache_match_with_llm(
            new_question="q", cached_question="q", cached_answer="a",
            timeout_sec=0.1,
        )
    assert got is False


# ---------------------------------------------------------------------------
# Integration: tier-3 gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier3_skipped_when_similarity_above_trust_threshold():
    """sim >= 0.98 → return cached row without calling the verifier."""
    stub, _ = _make_repo_query_stub(candidates=1, similarity=0.985)
    verifier = AsyncMock(return_value=False)  # would reject if called

    async def fake_embed(_q):
        return [0.1] * 8

    with (
        patch("open_notebook.domain.chat_cache.repo_query", new=stub),
        patch("open_notebook.utils.embedding.generate_embedding", new=fake_embed),
        patch(
            "open_notebook.domain.chat_cache._verify_cache_match_with_llm",
            new=verifier,
        ),
    ):
        result = await chat_cache.find_cached_answer(
            question="How do I use feature X on macOS?",
            context_fingerprint="fp",
            notebook_id="notebook:nb1",
        )
    assert result is not None and result["answer"] == "Cached answer body."
    verifier.assert_not_called()


@pytest.mark.asyncio
async def test_tier3_serves_cache_when_llm_approves():
    """sim in [0.92, 0.98) → verifier runs → YES → cache served."""
    stub, _ = _make_repo_query_stub(candidates=1, similarity=0.94)
    verifier = AsyncMock(return_value=True)

    async def fake_embed(_q):
        return [0.1] * 8

    with (
        patch("open_notebook.domain.chat_cache.repo_query", new=stub),
        patch("open_notebook.utils.embedding.generate_embedding", new=fake_embed),
        patch(
            "open_notebook.domain.chat_cache._verify_cache_match_with_llm",
            new=verifier,
        ),
    ):
        result = await chat_cache.find_cached_answer(
            question="How do I use feature X on macOS?",
            context_fingerprint="fp",
            notebook_id="notebook:nb1",
        )
    assert result is not None and result["answer"] == "Cached answer body."
    verifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_tier3_rejects_cache_when_llm_disapproves():
    """sim in danger band → verifier says NO → treated as miss (None)."""
    stub, _ = _make_repo_query_stub(candidates=1, similarity=0.93)
    verifier = AsyncMock(return_value=False)

    async def fake_embed(_q):
        return [0.1] * 8

    with (
        patch("open_notebook.domain.chat_cache.repo_query", new=stub),
        patch("open_notebook.utils.embedding.generate_embedding", new=fake_embed),
        patch(
            "open_notebook.domain.chat_cache._verify_cache_match_with_llm",
            new=verifier,
        ),
    ):
        result = await chat_cache.find_cached_answer(
            question="How do I use feature X on macOS?",
            context_fingerprint="fp",
            notebook_id="notebook:nb1",
        )
    assert result is None
    verifier.assert_awaited_once()


@pytest.mark.asyncio
async def test_tier3_disabled_flag_bypasses_verifier():
    """verify_semantic_match=False restores the pre-task behavior (sim ≥ threshold → hit)."""
    stub, _ = _make_repo_query_stub(candidates=1, similarity=0.93)
    verifier = AsyncMock(return_value=False)  # would reject if called

    async def fake_embed(_q):
        return [0.1] * 8

    with (
        patch("open_notebook.domain.chat_cache.repo_query", new=stub),
        patch("open_notebook.utils.embedding.generate_embedding", new=fake_embed),
        patch(
            "open_notebook.domain.chat_cache._verify_cache_match_with_llm",
            new=verifier,
        ),
    ):
        result = await chat_cache.find_cached_answer(
            question="q",
            context_fingerprint="fp",
            notebook_id="notebook:nb1",
            verify_semantic_match=False,
        )
    assert result is not None and result["answer"] == "Cached answer body."
    verifier.assert_not_called()


@pytest.mark.asyncio
async def test_below_min_similarity_is_plain_miss():
    """Below the floor we never reach verification."""
    stub, _ = _make_repo_query_stub(candidates=1, similarity=0.80)
    verifier = AsyncMock(return_value=True)

    async def fake_embed(_q):
        return [0.1] * 8

    with (
        patch("open_notebook.domain.chat_cache.repo_query", new=stub),
        patch("open_notebook.utils.embedding.generate_embedding", new=fake_embed),
        patch(
            "open_notebook.domain.chat_cache._verify_cache_match_with_llm",
            new=verifier,
        ),
    ):
        result = await chat_cache.find_cached_answer(
            question="q",
            context_fingerprint="fp",
            notebook_id="notebook:nb1",
        )
    assert result is None
    verifier.assert_not_called()
