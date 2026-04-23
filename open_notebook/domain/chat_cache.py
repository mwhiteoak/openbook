"""
Chat Q&A cache — persistent lookup of previously-answered questions.

When a user asks a question inside a notebook or source chat, we compute a
**context fingerprint** from the selected sources/notes, their inclusion
levels, the chat model, and the max `source.updated` timestamp. If a row
already exists for (scope, fingerprint, normalized question) we short-circuit
and stream the cached answer back instead of re-invoking the LLM.

Matching is three-tier:

1. **Exact match** — normalized question text (lowercased, whitespace
   collapsed). Powered by the composite indexes on
   ``(notebook_id|source_id, context_fingerprint, question_norm)``. This
   catches the 99% case: user re-opens the chat and asks the same question
   verbatim.

2. **Semantic match** — if the exact lookup misses, we fetch all rows for
   the current scope+fingerprint, re-embed the incoming question, and rank
   by cosine similarity against the stored question embeddings. A hit
   requires ``>= min_similarity`` (default 0.92 — high enough that cache
   hits only occur for genuinely the same question, not merely related
   ones). Brute-force per-scope is fine because the candidate set is small
   (dozens per notebook, not millions).

3. **LLM verification** — cosine similarity is a proxy, not a guarantee.
   Questions can score ≥0.92 yet demand materially different answers
   ("What *is* X?" vs "What *was* X?"; "How to configure X on Linux" vs
   "…on macOS"). For semantic hits below ``high_confidence_similarity``
   (default 0.98) we ask a cheap chat model to verify the cached answer
   actually addresses the new question. On YES → serve cached; on NO,
   timeout, or any error → fall through to a fresh LLM call.

   Verification adds ~200-800ms of latency but only when we'd otherwise
   risk serving a wrong answer. Above the 0.98 trust threshold we skip it
   entirely — the questions are essentially the same.

TTL and staleness are layered:
  * ``max_age_days`` (default 7) — rows older than this are ignored on
    read. Old entries remain in the table; a background cleanup job (future
    work) can delete by ``idx_chat_qa_created``.
  * Fingerprint change auto-invalidates — if a source is re-ingested or a
    different model is chosen, the fingerprint differs and the old rows
    are untouched by lookups (effectively invisible).

The table is schemaful (see migration 17) so we can rely on field types.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Dict, List, Optional

from loguru import logger
from pydantic import Field

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel
from open_notebook.exceptions import DatabaseOperationError
from open_notebook.utils.cache_fingerprint import normalize_question

# ---------------------------------------------------------------------------
# LLM verification tier config
#
# Tunable via env so ops can disable it without a code change if the chat
# model is flaky or the extra latency is unacceptable. Defaults keep the
# verifier ON — it's cheap (tens of tokens) and only fires on the narrow
# band between "probable match" and "obvious match".
# ---------------------------------------------------------------------------
_LLM_VERIFY_ENABLED_DEFAULT: bool = os.getenv(
    "OPEN_NOTEBOOK_CACHE_LLM_VERIFY", "1"
).strip().lower() not in {"0", "false", "no", "off", ""}

# Timeout for the YES/NO verification call. We bias *short* — the whole point
# is to avoid serving a wrong answer, not to win a performance lottery. On
# timeout we treat the cache as a miss (the user will get a fresh LLM call,
# which is what they wanted anyway).
_LLM_VERIFY_TIMEOUT_SEC: float = float(
    os.getenv("OPEN_NOTEBOOK_CACHE_LLM_VERIFY_TIMEOUT", "3.0")
)


class ChatQACache(ObjectModel):
    """A single cached answer.

    Scope is exactly one of ``notebook_id`` or ``source_id`` — the
    application enforces this; the schema allows both nullable for
    simplicity. ``context_fingerprint`` encodes everything else the LLM
    saw when producing the answer.
    """

    table_name: ClassVar[str] = "chat_qa_cache"
    # question_embedding is expected to be optional (may be None if the
    # embedding model was unavailable at write time); we still want it
    # persisted as None rather than omitted so the column shape is stable.
    nullable_fields: ClassVar[set[str]] = {
        "question_embedding",
        "notebook_id",
        "source_id",
        "model_id",
        "last_hit_at",
    }

    question: str
    question_norm: str
    question_embedding: Optional[List[float]] = None
    answer: str

    notebook_id: Optional[str] = None
    source_id: Optional[str] = None

    context_fingerprint: str
    model_id: Optional[str] = None

    hit_count: int = 0
    last_hit_at: Optional[datetime] = None


# ----------------------------------------------------------------------------
# Lookup / save helpers
#
# Kept as module-level async functions (rather than classmethods) for clarity
# in the router — they read/write their own rows and don't need instance
# state. Each function handles its own DB errors and logs rather than raising,
# because the chat flow must never be blocked by cache failures.
# ----------------------------------------------------------------------------


async def _verify_cache_match_with_llm(
    *,
    new_question: str,
    cached_question: str,
    cached_answer: str,
    model_id: Optional[str] = None,
    timeout_sec: float = _LLM_VERIFY_TIMEOUT_SEC,
) -> bool:
    """Ask a cheap LLM whether a cached answer truly addresses a new question.

    Returns True only if the model replies with a clear YES. Any other
    outcome — NO, timeout, provisioning failure, unexpected output — is
    treated as "do not trust the cache" and returns False. That asymmetry
    is deliberate: false positives (serving the wrong cached answer) are
    much worse for users than false negatives (extra LLM call they already
    expected).

    Kept out of ``find_cached_answer`` so it's easy to mock in tests.
    """
    # Import lazily — this module is imported at app startup and we don't
    # want a heavy AI-provisioning import path to run just to read a cache
    # entry that's going to exact-match 99% of the time.
    try:
        from open_notebook.ai.provision import provision_langchain_model
    except Exception as e:  # pragma: no cover — import-time breakage is fatal elsewhere
        logger.debug(f"cache-verify: provisioning import failed ({e}); skipping")
        return False

    # Compact prompt — we charge tokens for every verification, so keep this
    # tight. The single-line answer format makes the YES/NO parse robust to
    # extra whitespace / capitalization without needing regex.
    prompt = (
        "You are verifying whether a previously cached answer can be reused.\n"
        "\n"
        f"NEW_QUESTION: {new_question.strip()}\n"
        f"CACHED_QUESTION: {cached_question.strip()}\n"
        f"CACHED_ANSWER: {cached_answer.strip()[:1500]}\n"
        "\n"
        "Would the CACHED_ANSWER correctly and completely answer the "
        "NEW_QUESTION? Treat changes in scope, tense, entity, location, "
        "version, or platform as disqualifying.\n"
        "\n"
        "Reply with exactly one word: YES or NO."
    )

    try:
        model = await provision_langchain_model(
            prompt, model_id, "chat", max_tokens=5
        )
        result = await asyncio.wait_for(model.ainvoke(prompt), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.debug("cache-verify: LLM timed out; treating as miss")
        return False
    except Exception as e:
        logger.debug(f"cache-verify: LLM call failed ({e}); treating as miss")
        return False

    # extract_text_content + clean_thinking_content live in the graphs
    # module — import lazily to avoid circular imports at module load.
    try:
        from open_notebook.utils.text_utils import (
            clean_thinking_content,
            extract_text_content,
        )
        raw = clean_thinking_content(extract_text_content(result.content))
    except Exception:
        # Fallback: LangChain content is usually a plain str; tolerate the
        # rare dict/list shape by coercing to str.
        raw = str(getattr(result, "content", result))

    answer = raw.strip().upper()
    # Accept the first token — some models prepend a period or trailing
    # punctuation ("YES.", "YES\n"). Reject anything that doesn't *start*
    # with YES, because "NO, but…" must not count as a YES.
    verdict = answer.startswith("YES")
    logger.debug(f"cache-verify: model replied {answer[:20]!r} → verified={verdict}")
    return verdict


async def find_cached_answer(
    *,
    question: str,
    context_fingerprint: str,
    notebook_id: Optional[str] = None,
    source_id: Optional[str] = None,
    model_id: Optional[str] = None,
    max_age_days: int = 7,
    min_similarity: float = 0.92,
    high_confidence_similarity: float = 0.98,
    verify_semantic_match: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Look up a cached answer; returns the row as a dict or None on miss.

    Returns a dict with at least ``id``, ``question``, ``answer``,
    ``hit_count``, ``created`` on hit. The caller typically only needs
    ``answer`` + ``id`` (for the follow-up hit-count bump) but we pass the
    full row through in case we want to surface cached-at timestamps in the
    UI later.

    Args:
        question: The raw user question.
        context_fingerprint: Hash from ``compute_context_fingerprint``.
        notebook_id / source_id: Exactly one should be set. Identifies the
            chat scope.
        model_id: Included in the stored row for debugging / analytics; not
            used for matching (already baked into the fingerprint). Also
            passed to the verification LLM so users on small local models
            don't get hit with a sudden GPT-4 call.
        max_age_days: Ignore cache rows older than this many days.
        min_similarity: Minimum cosine similarity for a semantic fallback
            hit to be *considered*. Between this and
            ``high_confidence_similarity``, the LLM verification tier gates
            the hit.
        high_confidence_similarity: Similarity at/above which we trust the
            semantic match without LLM verification (default 0.98). This
            avoids the verification cost on the very-likely matches while
            still catching the dangerous "similar but different" band.
        verify_semantic_match: Whether to run LLM verification on hits
            below ``high_confidence_similarity``. ``None`` → use the
            ``OPEN_NOTEBOOK_CACHE_LLM_VERIFY`` env default (on).
    """
    if not notebook_id and not source_id:
        logger.debug("Cache lookup called without a scope; skipping")
        return None

    # Both set is a programming error at the call site — be loud about it.
    if notebook_id and source_id:
        logger.warning(
            "Cache lookup received both notebook_id and source_id; using notebook scope"
        )
        source_id = None

    q_norm = normalize_question(question)
    if not q_norm:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    # ---- Tier 1: exact normalized match ---------------------------------
    try:
        if notebook_id:
            exact_query = """
                SELECT * FROM chat_qa_cache
                WHERE notebook_id = $scope
                  AND context_fingerprint = $fp
                  AND question_norm = $q_norm
                  AND created >= $cutoff
                ORDER BY created DESC
                LIMIT 1
            """
            params = {
                "scope": ensure_record_id(notebook_id),
                "fp": context_fingerprint,
                "q_norm": q_norm,
                "cutoff": cutoff,
            }
        else:
            assert source_id is not None
            exact_query = """
                SELECT * FROM chat_qa_cache
                WHERE source_id = $scope
                  AND context_fingerprint = $fp
                  AND question_norm = $q_norm
                  AND created >= $cutoff
                ORDER BY created DESC
                LIMIT 1
            """
            params = {
                "scope": ensure_record_id(source_id),
                "fp": context_fingerprint,
                "q_norm": q_norm,
                "cutoff": cutoff,
            }

        exact = await repo_query(exact_query, params)
        if exact:
            row = exact[0]
            logger.debug(
                f"Chat cache exact-hit on question_norm (scope={'notebook' if notebook_id else 'source'})"
            )
            return _normalize_row(row)
    except Exception as e:
        # Cache failures must never break the chat flow — log and fall through.
        logger.warning(f"Chat cache exact lookup failed: {e}")

    # ---- Tier 2: semantic fallback --------------------------------------
    # Generate an embedding for the incoming question and rank the scoped
    # candidates by cosine similarity. We do this in one query using
    # vector::similarity::cosine() directly against the stored embeddings.
    #
    # IMPORTANT: short-circuit before embedding if there are no candidate
    # rows with embeddings for this scope+fingerprint. Generating an
    # embedding is an LLM round-trip (50-2000ms depending on provider); we
    # absolutely do not want to pay that on every cache miss — especially on
    # brand-new notebooks where the table is empty. One cheap COUNT query
    # avoids that.
    try:
        if notebook_id:
            probe_query = """
                SELECT count() AS n FROM chat_qa_cache
                WHERE notebook_id = $scope
                  AND context_fingerprint = $fp
                  AND created >= $cutoff
                  AND question_embedding IS NOT NONE
                GROUP ALL
                LIMIT 1
            """
            probe_params = {
                "scope": ensure_record_id(notebook_id),
                "fp": context_fingerprint,
                "cutoff": cutoff,
            }
        else:
            assert source_id is not None
            probe_query = """
                SELECT count() AS n FROM chat_qa_cache
                WHERE source_id = $scope
                  AND context_fingerprint = $fp
                  AND created >= $cutoff
                  AND question_embedding IS NOT NONE
                GROUP ALL
                LIMIT 1
            """
            probe_params = {
                "scope": ensure_record_id(source_id),
                "fp": context_fingerprint,
                "cutoff": cutoff,
            }

        probe = await repo_query(probe_query, probe_params)
        candidate_count = int(probe[0].get("n", 0)) if probe else 0
        if candidate_count == 0:
            # Nothing to compare against — don't waste an embedding round-trip.
            return None
    except Exception as e:
        # If the probe fails, bail rather than falling back to the expensive
        # path. Exact-match still handled the 99% case above.
        logger.debug(f"Chat cache candidate probe failed, skipping semantic tier: {e}")
        return None

    try:
        from open_notebook.utils.embedding import generate_embedding

        q_embed = await generate_embedding(question)
    except Exception as e:
        # No embedding model configured, or embedding call failed — skip the
        # semantic tier rather than blowing up. Exact-match-only is still
        # useful.
        logger.debug(f"Chat cache semantic fallback skipped (no embedding): {e}")
        return None

    try:
        if notebook_id:
            semantic_query = """
                SELECT *, vector::similarity::cosine(question_embedding, $embed) AS similarity
                FROM chat_qa_cache
                WHERE notebook_id = $scope
                  AND context_fingerprint = $fp
                  AND created >= $cutoff
                  AND question_embedding IS NOT NONE
                ORDER BY similarity DESC
                LIMIT 1
            """
            params = {
                "scope": ensure_record_id(notebook_id),
                "fp": context_fingerprint,
                "cutoff": cutoff,
                "embed": q_embed,
            }
        else:
            assert source_id is not None
            semantic_query = """
                SELECT *, vector::similarity::cosine(question_embedding, $embed) AS similarity
                FROM chat_qa_cache
                WHERE source_id = $scope
                  AND context_fingerprint = $fp
                  AND created >= $cutoff
                  AND question_embedding IS NOT NONE
                ORDER BY similarity DESC
                LIMIT 1
            """
            params = {
                "scope": ensure_record_id(source_id),
                "fp": context_fingerprint,
                "cutoff": cutoff,
                "embed": q_embed,
            }

        rows = await repo_query(semantic_query, params)
        if rows:
            best = rows[0]
            similarity = float(best.get("similarity") or 0.0)
            if similarity < min_similarity:
                logger.debug(
                    f"Chat cache semantic-miss (best similarity={similarity:.3f} < {min_similarity})"
                )
                return None

            # Tier 3: above the trust threshold we skip verification —
            # embeddings that close are effectively the same question.
            if similarity >= high_confidence_similarity:
                logger.debug(
                    f"Chat cache semantic-hit (high-confidence, similarity={similarity:.3f})"
                )
                return _normalize_row(best)

            # In the danger band — verify with a cheap LLM before serving.
            use_verifier = (
                verify_semantic_match
                if verify_semantic_match is not None
                else _LLM_VERIFY_ENABLED_DEFAULT
            )
            if not use_verifier:
                logger.debug(
                    f"Chat cache semantic-hit (verifier disabled, similarity={similarity:.3f})"
                )
                return _normalize_row(best)

            cached_q = str(best.get("question") or "")
            cached_a = str(best.get("answer") or "")
            verified = await _verify_cache_match_with_llm(
                new_question=question,
                cached_question=cached_q,
                cached_answer=cached_a,
                model_id=model_id,
            )
            if verified:
                logger.debug(
                    f"Chat cache semantic-hit (llm-verified, similarity={similarity:.3f})"
                )
                return _normalize_row(best)
            logger.debug(
                f"Chat cache semantic-miss (llm rejected, similarity={similarity:.3f})"
            )
    except Exception as e:
        logger.warning(f"Chat cache semantic lookup failed: {e}")

    return None


async def save_cached_answer(
    *,
    question: str,
    answer: str,
    context_fingerprint: str,
    notebook_id: Optional[str] = None,
    source_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Optional[str]:
    """Persist a newly-generated Q&A pair to the cache.

    Returns the record id, or None if save fails (we swallow and log —
    cache writes are fire-and-forget; the chat response has already been
    streamed to the user).
    """
    if not notebook_id and not source_id:
        return None
    if not question.strip() or not answer.strip():
        return None

    q_norm = normalize_question(question)

    # Best-effort embed — if it fails, store the row without an embedding so
    # future lookups can still find it via exact match. This matters for the
    # user's very first question; we don't want a transient embedding outage
    # to disable the cache entirely.
    q_embed: Optional[List[float]] = None
    try:
        from open_notebook.utils.embedding import generate_embedding

        q_embed = await generate_embedding(question)
    except Exception as e:
        logger.debug(f"Cache write: skipping embedding generation: {e}")

    try:
        entry = ChatQACache(
            question=question.strip(),
            question_norm=q_norm,
            question_embedding=q_embed,
            answer=answer,
            notebook_id=notebook_id,
            source_id=source_id,
            context_fingerprint=context_fingerprint,
            model_id=model_id,
            hit_count=0,
        )
        await entry.save()
        return entry.id
    except Exception as e:
        logger.warning(f"Chat cache save failed: {e}")
        return None


async def bump_cache_hit(cache_id: str) -> None:
    """Increment hit counter and update last_hit_at. Fire-and-forget.

    Runs in a single UPDATE statement so concurrent hits on the same row
    don't lose count (SurrealDB serializes table writes per record).
    """
    if not cache_id:
        return
    try:
        await repo_query(
            """
            UPDATE $id SET
                hit_count = (hit_count OR 0) + 1,
                last_hit_at = time::now()
            """,
            {"id": ensure_record_id(cache_id)},
        )
    except Exception as e:
        # Stats are nice-to-have; do not block on failure.
        logger.debug(f"Chat cache hit-bump failed for {cache_id}: {e}")


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten SurrealDB RecordIDs to strings in the cache row dict.

    The chat router needs plain strings for the SSE payload (JSON-safe)
    and for the follow-up bump call.
    """
    from open_notebook.database.repository import parse_record_ids

    return parse_record_ids(row)


# Expose a narrow public API from this module.
__all__ = [
    "ChatQACache",
    "find_cached_answer",
    "save_cached_answer",
    "bump_cache_hit",
    # Exposed so integration tests can monkeypatch it — not part of the
    # router-facing API. Treat as internal.
    "_verify_cache_match_with_llm",
]
