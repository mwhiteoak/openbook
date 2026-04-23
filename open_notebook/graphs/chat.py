import asyncio
import os
import re
from typing import Annotated, Optional

import aiosqlite
from ai_prompter import Prompter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.database.repository import repo_query
from open_notebook.domain.notebook import Notebook, text_search, vector_search
from open_notebook.exceptions import OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content

# Cap for retrieval-augmented excerpts appended to the LLM context. These
# numbers are a deliberately conservative starting point — high enough to
# surface useful passages for most questions, low enough to keep token bloat
# bounded. Tune if users report "model still can't find it" on specific
# questions (raise RETRIEVAL_TOP_K) or runaway cost (lower MAX_CHARS_PER_CHUNK).
RETRIEVAL_TOP_K = 6
RETRIEVAL_MIN_SCORE = 0.25  # default is 0.2 — bump slightly to filter noise
MAX_CHARS_PER_CHUNK = 1200

# Reciprocal Rank Fusion (RRF) constant. 60 is the canonical value from
# Cormack et al. (2009) and is used by Elastic/Weaviate/pgvector examples.
# Smaller k gives more weight to top-ranked items from each list; larger k
# flattens the distribution. 60 is a safe default — don't tweak unless you're
# measuring retrieval quality on a real benchmark.
RRF_K = 60

# Reranking. RRF gives us a robust ordering using two lexical/semantic signals
# but has no semantic awareness of the question's actual *meaning* — it only
# knows where each source ranked in the two lists. A cross-encoder or LLM
# rerank can re-order the top-N candidates using a fuller relevance judgment
# (does this passage actually answer the question?).
#
# We over-fetch RERANK_CANDIDATES, let an LLM rerank them against the question,
# then keep the top RETRIEVAL_TOP_K. If the LLM call fails or the rerank
# returns garbage, we fall back to the RRF order — rerank must never be worse
# than the baseline.
RERANK_CANDIDATES = 20
RERANK_ENABLED = True
RERANK_SNIPPET_CHARS = 500  # chars shown to the reranker per candidate


# --- Smart model routing ---------------------------------------------------
#
# Not every question deserves GPT-4o-equivalent. Short factual lookups ("what
# year was X founded?", "summarize source Y") complete fine on a small/cheap
# model and cost ~10x less. Only the genuinely hard questions — long
# reasoning, multi-document synthesis, coding — need the flagship.
#
# The router picks between two *configured* model IDs set via env:
#
#   OPEN_NOTEBOOK_ROUTER_CHEAP_MODEL  — e.g. "gpt-4o-mini" or a local 8B
#   OPEN_NOTEBOOK_ROUTER_STRONG_MODEL — e.g. "gpt-4o" or "claude-sonnet"
#
# If either env is missing the router is a no-op — we fall back to whatever
# model_id the caller already resolved (session override, app default, or
# provision_langchain_model's internal defaulting). This keeps the feature
# opt-in and safe to ship off-by-default.
#
# Routing signals (any one promotes to strong):
#   * question length >= ROUTER_LONG_QUESTION_CHARS
#   * question matches ROUTER_REASONING_PATTERN (explain/compare/analyze/…)
#   * retrieved excerpt count >= ROUTER_MULTI_SOURCE_THRESHOLD
#   * rendered system prompt >= ROUTER_LARGE_CONTEXT_CHARS
#
# We deliberately bias toward "strong" — false positives (using the expensive
# model on an easy question) waste money, but false negatives (cheap model
# on a hard question) give the user a bad answer. Users with an explicit
# model_override always bypass routing; the UI toggle is sacred.

_ROUTER_CHEAP_MODEL: Optional[str] = (
    os.getenv("OPEN_NOTEBOOK_ROUTER_CHEAP_MODEL") or None
)
_ROUTER_STRONG_MODEL: Optional[str] = (
    os.getenv("OPEN_NOTEBOOK_ROUTER_STRONG_MODEL") or None
)
_ROUTER_ENABLED: bool = (
    bool(_ROUTER_CHEAP_MODEL)
    and bool(_ROUTER_STRONG_MODEL)
    and os.getenv("OPEN_NOTEBOOK_ROUTER_ENABLED", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)

ROUTER_LONG_QUESTION_CHARS = 220
ROUTER_MULTI_SOURCE_THRESHOLD = 4
ROUTER_LARGE_CONTEXT_CHARS = 12_000

# Match at word boundary so "explainer" doesn't trigger on "explain". The
# keywords lean on reasoning/synthesis intent — not mere question length.
_ROUTER_REASONING_PATTERN = re.compile(
    r"\b(explain|compare|contrast|analyz(e|ing)|evaluat(e|ing)|"
    r"why|trade[- ]?off|pros and cons|implication|critique|synthesi[sz]e|"
    r"design|derive|prove|debate|argue)\b",
    re.IGNORECASE,
)


def _route_model(
    *,
    question: str,
    context_chars: int,
    excerpt_count: int,
) -> Optional[str]:
    """Pick a model_id based on question difficulty signals.

    Returns ``_ROUTER_STRONG_MODEL`` if any signal fires, otherwise
    ``_ROUTER_CHEAP_MODEL``. Returns ``None`` when routing is disabled
    (env vars missing) so the caller keeps its existing choice.

    Pure function — no side effects, trivially unit-testable.
    """
    if not _ROUTER_ENABLED:
        return None

    q = (question or "").strip()
    if not q:
        # Empty/whitespace — cheapest wins by default.
        return _ROUTER_CHEAP_MODEL

    if len(q) >= ROUTER_LONG_QUESTION_CHARS:
        return _ROUTER_STRONG_MODEL
    if _ROUTER_REASONING_PATTERN.search(q):
        return _ROUTER_STRONG_MODEL
    if excerpt_count >= ROUTER_MULTI_SOURCE_THRESHOLD:
        return _ROUTER_STRONG_MODEL
    if context_chars >= ROUTER_LARGE_CONTEXT_CHARS:
        return _ROUTER_STRONG_MODEL

    return _ROUTER_CHEAP_MODEL


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]
    # Retrieval preview — populated by the ``retrieve`` node with a light-weight
    # description of the excerpts about to feed the LLM. Consumers (the SSE
    # streamer in api/routers/chat.py) forward this to the frontend BEFORE the
    # answer tokens start flowing so the user immediately sees "Searching 3
    # sources…" Perplexity-style. ``None`` when retrieval was a no-op.
    retrieval_preview: Optional[list]


# --- Follow-up aware query rewriting --------------------------------------
#
# Why this matters:
#   User turn 1: "Explain the Baxter protocol."
#   Assistant:   <long answer about the Baxter protocol>
#   User turn 2: "What about its failure modes?"
#
# If we feed "What about its failure modes?" verbatim into hybrid retrieval,
# we get back chunks about "failure modes" in general — anything in the
# notebook that mentions "failure" ranks well, but the specific Baxter-protocol
# content we actually want probably doesn't, because its chunks don't restate
# the phrase. The retrieval quality collapses.
#
# Fix: before retrieval, rewrite the current user question into a standalone
# query using the prior conversation. "What about its failure modes?" becomes
# "What are the failure modes of the Baxter protocol?" — and now hybrid
# retrieval has a fighting chance.
#
# This costs one small LLM call per follow-up turn. To keep that cost in check
# we gate the call behind a cheap heuristic: only rewrite when the question
# LOOKS like it might need context (short, or starts with a typical follow-up
# marker, or leads with a pronoun). A self-contained question like
# "What is reinforcement learning?" skips the rewrite entirely.
# Markers that only count as follow-up signals when they appear at the START
# of the question. "And why?" is a follow-up; "pros and cons" is not.
_FOLLOWUP_STARTS = (
    "and ", "also ", "more ", "why ", "why not", "how about ", "what about ",
    "what else", "any other ", "the other ", "same for ", "tell me more",
    "elaborate", "expand on ", "continue", "go on", "explain that",
)
_FOLLOWUP_LEAD_PRONOUNS = (
    "it ", "its ", "it's ", "they ", "their ", "them ",
    "that ", "this ", "those ", "these ", "he ", "she ", "his ", "her ",
)


def _looks_like_followup(question: str) -> bool:
    """Cheap heuristic for whether ``question`` likely needs context to resolve.

    False positives are fine (we'll just do an unnecessary rewrite).
    False negatives are the real cost (we'll retrieve with a bad query).
    Stay generous.
    """
    q = question.strip().lower()
    if not q:
        return False
    # Very short questions are almost certainly follow-ups.
    if len(q) <= 40:
        return True
    if any(q.startswith(p) for p in _FOLLOWUP_LEAD_PRONOUNS):
        return True
    if any(q.startswith(marker) for marker in _FOLLOWUP_STARTS):
        return True
    return False


async def _rewrite_followup_question(
    question: str,
    messages: list,
    config: RunnableConfig,
    state: ThreadState,
) -> str:
    """Return a standalone version of ``question`` if it looks like a follow-up.

    Returns the original question unchanged when:
      * there's no prior assistant turn (nothing to resolve against);
      * the heuristic says the question looks self-contained;
      * the rewrite LLM call fails for any reason.
    """
    # Need at least one prior assistant message to anchor pronouns against.
    prior = [m for m in messages[:-1] if getattr(m, "type", None) in ("human", "ai")]
    if not any(getattr(m, "type", None) == "ai" for m in prior):
        return question
    if not _looks_like_followup(question):
        return question

    # Keep only the most recent turn pair — enough context to disambiguate
    # pronouns without burning tokens on old topics the user has moved past.
    recent_pairs: list[tuple[str, str]] = []
    last_human: Optional[str] = None
    for m in prior[-6:]:  # up to ~3 turn-pairs
        text = extract_text_content(m.content).strip()
        if not text:
            continue
        if getattr(m, "type", None) == "human":
            last_human = text
        elif getattr(m, "type", None) == "ai" and last_human is not None:
            recent_pairs.append((last_human, text))
            last_human = None

    if not recent_pairs:
        return question

    # Truncate the assistant replies — a long answer will dominate the context
    # window for an unrelated reformulation call. 600 chars keeps pronoun
    # antecedents intact without bloat.
    history_lines: list[str] = []
    for h, a in recent_pairs[-2:]:
        history_lines.append(f"User: {h}")
        history_lines.append(f"Assistant: {a[:600]}")
    history_block = "\n".join(history_lines)

    rewrite_prompt = (
        "You rewrite follow-up questions into standalone search queries.\n\n"
        "Given the conversation below, rewrite the LAST user question so it can "
        "be understood on its own — resolving pronouns, implicit subjects, and "
        "references to prior turns. Keep it short and keyword-rich. If the "
        "question is already self-contained, return it unchanged.\n\n"
        "Return ONLY the rewritten question. No preamble, no quotes, no "
        "explanation. If unsure, return the original question verbatim.\n\n"
        f"CONVERSATION:\n{history_block}\n\n"
        f"LAST USER QUESTION: {question}\n\n"
        "REWRITTEN QUESTION:"
    )

    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )
        # Use the chat model (falls back to default). Keep max_tokens small —
        # a rewrite is one sentence; anything more is the model rambling.
        model = await provision_langchain_model(
            rewrite_prompt, model_id, "chat", max_tokens=120
        )
        result = await asyncio.wait_for(model.ainvoke(rewrite_prompt), timeout=8.0)
        rewritten = extract_text_content(result.content).strip()
        rewritten = clean_thinking_content(rewritten)
        # Strip common wrapping artifacts.
        rewritten = rewritten.strip().strip('"').strip("'").strip()
        # Sanity check: reject empty or suspicious rewrites (e.g. the model
        # returned a full paragraph instead of a question).
        if not rewritten:
            return question
        if len(rewritten) > 400:
            logger.debug(
                f"followup rewrite: discarded over-long result ({len(rewritten)} chars)"
            )
            return question
        if rewritten.lower() == question.strip().lower():
            return question
        logger.debug(
            f"followup rewrite: '{question[:60]}…' → '{rewritten[:60]}…'"
        )
        return rewritten
    except Exception as e:
        logger.debug(f"followup rewrite: suppressed error: {e}")
        return question


async def _rerank_candidates(
    question: str,
    candidates: list[dict],
    config: RunnableConfig,
    state: ThreadState,
) -> list[int]:
    """Return candidate indices in reranked (best-first) order.

    The reranker sees the question and a numbered list of short excerpt
    snippets, and returns a comma-separated list of indices (1-based in the
    prompt, converted to 0-based in the return value). Irrelevant candidates
    can be omitted by the model — they'll be appended at the end in their
    original RRF order so we never lose them entirely if the reranker
    over-filters.

    On any failure (timeout, parse error, empty response), returns
    ``list(range(len(candidates)))`` unchanged. Rerank must never regress
    below RRF baseline.
    """
    if not RERANK_ENABLED or len(candidates) <= 1:
        return list(range(len(candidates)))

    # Build a compact numbered listing. We deliberately keep snippets short
    # here — the reranker is judging relevance, not generating an answer, so
    # 500 chars of leading content is plenty. Keeping the prompt small also
    # keeps rerank latency low.
    lines = [
        "You are a passage reranker. Given the user's question and a numbered",
        "list of passages, output ONLY a comma-separated list of passage",
        "numbers in best-to-worst order of relevance. Omit clearly irrelevant",
        "passages. Do not include any other text.",
        "",
        f"QUESTION: {question}",
        "",
        "PASSAGES:",
    ]
    for i, c in enumerate(candidates, start=1):
        snippet = (c.get("content") or "")[:RERANK_SNIPPET_CHARS]
        snippet = snippet.replace("\n", " ").strip()
        lines.append(f"{i}. {snippet}")
    lines.append("")
    lines.append("RANKED ORDER (comma-separated numbers only):")
    prompt = "\n".join(lines)

    try:
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )
        model = await provision_langchain_model(
            prompt, model_id, "chat", max_tokens=120
        )
        result = await asyncio.wait_for(model.ainvoke(prompt), timeout=10.0)
        raw = extract_text_content(result.content)
        raw = clean_thinking_content(raw).strip()
    except Exception as e:
        logger.debug(f"rerank: LLM call failed, falling back to RRF: {e}")
        return list(range(len(candidates)))

    # Parse: accept digits separated by commas/whitespace. Be liberal — the
    # model might return "3, 1, 5" or "3,1,5" or "3 1 5".
    try:
        tokens = re.split(r"[^\d]+", raw)
        parsed = [int(t) - 1 for t in tokens if t and t.isdigit()]
    except Exception as e:
        logger.debug(f"rerank: parse failed ({e!r}), raw={raw[:80]!r}")
        return list(range(len(candidates)))

    n = len(candidates)
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in parsed:
        if 0 <= idx < n and idx not in seen:
            ordered.append(idx)
            seen.add(idx)

    if not ordered:
        logger.debug(f"rerank: empty parse, raw={raw[:80]!r}")
        return list(range(n))

    # Append any candidates the reranker omitted, preserving their original
    # RRF order. This is the safety net: if the model incorrectly dropped a
    # useful passage, we still surface it lower in the list rather than
    # losing it entirely.
    for i in range(n):
        if i not in seen:
            ordered.append(i)
    logger.debug(
        f"rerank: reordered {n} candidates; new top-3 indices = {ordered[:3]}"
    )
    return ordered


async def retrieve_from_embeddings(
    state: ThreadState, config: RunnableConfig
) -> dict:
    """Augment the chat context with semantic-search hits against source chunks.

    **Why this exists:** the user-facing context picker lets users toggle
    sources between ``insights`` (short summary) and ``full content``. When a
    user asks a question whose answer lives deep in a source body but they
    only selected the insights/summary for that source, the LLM previously
    responded "not in the context." That's bad UX — the information IS in
    the notebook, the summary just didn't surface it.

    **What we do:** take the user's latest question, run a vector search
    across all embedded source chunks in this notebook, and append the top
    hits to ``context`` under a clearly-labeled section. The LLM's system
    prompt (``prompts/chat/system.jinja``) instructs it to fall back to
    these excerpts when the primary context doesn't answer the question.

    **Scoping:** vector_search is global across the user's data, so we
    post-filter by the notebook's own source IDs to prevent leaking chunks
    from unrelated notebooks. This adds one DB round-trip (``get_sources``)
    but it's cheap and essential for correctness.

    **Failure mode:** retrieval is best-effort. Any error (no embedding
    model, SurrealDB hiccup, notebook without sources) is logged and
    swallowed — the chat flow proceeds with whatever context already exists.
    """
    try:
        notebook: Optional[Notebook] = state.get("notebook")
        if notebook is None:
            return {}

        # Find the latest human message to use as the retrieval query.
        messages = state.get("messages", []) or []
        last_human = next(
            (
                m
                for m in reversed(messages)
                if isinstance(m, HumanMessage)
            ),
            None,
        )
        if last_human is None:
            return {}
        raw_question = extract_text_content(last_human.content).strip()
        if not raw_question:
            return {}

        # Resolve pronouns/implicit context from prior turns into a standalone
        # query before hybrid retrieval runs. No-op for first turns and for
        # questions that already look self-contained. The LLM still sees the
        # ORIGINAL raw_question in the chat history — we only swap for
        # retrieval scoring purposes.
        question = await _rewrite_followup_question(
            raw_question, messages, config, state
        )

        # Scope: collect this notebook's source IDs so we can filter out
        # cross-notebook hits. An empty notebook (no sources) has no value to
        # retrieve from, so bail early to save the embedding round-trip.
        try:
            scoped_sources = await notebook.get_sources()
        except Exception as e:
            logger.debug(f"chat retrieval: failed to fetch notebook sources: {e}")
            return {}
        if not scoped_sources:
            return {}
        scoped_ids = {s.id for s in scoped_sources if getattr(s, "id", None)}
        # Title map for enriching the retrieval preview with human-readable
        # labels. The frontend uses this to render "Searching 3 sources" chips.
        title_by_source: dict[str, str] = {
            s.id: (getattr(s, "title", None) or "Untitled source")
            for s in scoped_sources
            if getattr(s, "id", None)
        }

        # --- Hybrid retrieval: vector + BM25/full-text, merged with RRF ----
        #
        # Why both:
        #   * Vector captures semantic similarity — good for paraphrased or
        #     conceptual questions ("what are the main arguments against X?")
        #   * Full-text captures lexical exact-match signal — good when the
        #     user remembers a specific term, proper noun, or quoted phrase
        #     that embeddings tend to blur ("the Baxter protocol", "GPT-4.1")
        #
        # We fan these out in parallel so the hybrid path costs one
        # max(vector, text) latency instead of summing them. If either call
        # fails, we fall back to the surviving one — hybrid should NEVER be
        # worse than the single-index baseline.
        #
        # We over-fetch enough to feed the reranker — the reranker's job is
        # to reorder within this pool, so we want a pool bigger than what we
        # ultimately keep.
        over_fetch = max(RETRIEVAL_TOP_K * 3, RERANK_CANDIDATES, 12)

        vector_task = asyncio.create_task(
            vector_search(
                keyword=question,
                results=over_fetch,
                source=True,
                note=False,  # notes already come via curated CONTEXT
                minimum_score=RETRIEVAL_MIN_SCORE,
            )
        )
        text_task = asyncio.create_task(
            text_search(
                keyword=question,
                results=over_fetch,
                source=True,
                note=False,
            )
        )

        vector_hits_raw, text_hits_raw = await asyncio.gather(
            vector_task, text_task, return_exceptions=True
        )
        if isinstance(vector_hits_raw, BaseException):
            logger.debug(f"chat retrieval: vector_search failed: {vector_hits_raw}")
            vector_hits: list[dict] = []
        else:
            vector_hits = list(vector_hits_raw or [])
        if isinstance(text_hits_raw, BaseException):
            logger.debug(f"chat retrieval: text_search failed: {text_hits_raw}")
            text_hits: list[dict] = []
        else:
            text_hits = list(text_hits_raw or [])

        if not vector_hits and not text_hits:
            return {}

        # --- Build source-level ranks -------------------------------------
        # vector_search is chunk-granular; text_search is source-granular.
        # We unify at the source level for RRF: each source's rank in a list
        # is the position of its FIRST appearance (which, since both lists
        # are pre-sorted by their respective scores, is the best-matching
        # row for that source).
        def _first_occurrence_ranks(hits: list[dict]) -> dict[str, int]:
            ranks: dict[str, int] = {}
            for i, h in enumerate(hits):
                sid = str(h.get("item_id") or h.get("id") or "")
                if sid and sid not in ranks:
                    ranks[sid] = i
            return ranks

        vector_ranks = _first_occurrence_ranks(vector_hits)
        text_ranks = _first_occurrence_ranks(text_hits)

        # Keep the best chunk per source (vector_hits first hit per source).
        # We'll use these as the excerpt content the LLM actually sees.
        best_chunk_by_source: dict[str, dict] = {}
        for h in vector_hits:
            sid = str(h.get("item_id") or h.get("id") or "")
            if not sid or sid in best_chunk_by_source:
                continue
            best_chunk_by_source[sid] = h

        # --- RRF merge ----------------------------------------------------
        # score(source) = sum over lists of 1 / (RRF_K + rank)
        # Missing from a list contributes 0 (not an error — just means the
        # other signal has to carry it).
        all_source_ids = set(vector_ranks) | set(text_ranks)
        rrf_scores: dict[str, float] = {}
        for sid in all_source_ids:
            score = 0.0
            if sid in vector_ranks:
                score += 1.0 / (RRF_K + vector_ranks[sid])
            if sid in text_ranks:
                score += 1.0 / (RRF_K + text_ranks[sid])
            rrf_scores[sid] = score

        ranked_source_ids = sorted(
            all_source_ids, key=lambda s: rrf_scores[s], reverse=True
        )

        # --- Build the rerank candidate pool ------------------------------
        # Walk the RRF-ordered source IDs and materialize up to
        # RERANK_CANDIDATES entries (bounded for prompt size). This is wider
        # than what we'll ultimately keep — the reranker's job is to
        # reshuffle within this pool.
        candidates: list[dict] = []
        seen: set[str] = set()
        for sid in ranked_source_ids:
            if sid not in scoped_ids:
                continue
            chunk = best_chunk_by_source.get(sid)
            # If a source only surfaced via full-text search (no embedding
            # hit at all), we won't have a chunk. Skip for now — the rare
            # case of "text-only match" can be revisited if users report it.
            if not chunk:
                continue
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue
            dedupe_key = sid + "::" + content[:120]
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append({
                "source_id": sid,
                "content": content,
                "similarity": float(chunk.get("similarity") or 0.0),
                "rrf_score": rrf_scores[sid],
                # Diagnostic flags — useful if we ever want to surface
                # "matched lexically AND semantically" in the UI.
                "matched_vector": sid in vector_ranks,
                "matched_text": sid in text_ranks,
            })
            if len(candidates) >= RERANK_CANDIDATES:
                break

        if not candidates:
            return {}

        # --- Rerank and trim to final top-K -------------------------------
        # The reranker reorders the pool using the full question. On failure
        # it returns the identity ordering so we never regress below RRF.
        rerank_order = await _rerank_candidates(question, candidates, config, state)
        kept = [candidates[i] for i in rerank_order[:RETRIEVAL_TOP_K]]
        if not kept:
            # Shouldn't happen since candidates is non-empty and rerank
            # returns at least identity, but belt-and-suspenders.
            kept = candidates[:RETRIEVAL_TOP_K]

        # Render as a Markdown-ish block the LLM can consume alongside the
        # existing context. Source IDs are included verbatim because the
        # chat/system.jinja citation rules require [source:id] references.
        lines = [
            "",
            "# DEEPER SEARCH EXCERPTS",
            "",
            (
                "The following passages were retrieved by semantic search "
                "against this notebook's embedded source chunks because the "
                "user's question may not be answered by the summaries above. "
                "Cite them using their source IDs when you draw on them."
            ),
            "",
        ]
        for i, h in enumerate(kept, start=1):
            snippet = h["content"]
            if len(snippet) > MAX_CHARS_PER_CHUNK:
                snippet = snippet[:MAX_CHARS_PER_CHUNK].rstrip() + "…"
            lines.append(
                f"## Excerpt {i} — similarity {h['similarity']:.2f} — [{h['source_id']}]"
            )
            lines.append("")
            lines.append(snippet)
            lines.append("")

        extra = "\n".join(lines)
        existing = state.get("context") or ""
        merged = (existing.rstrip() + "\n\n" + extra) if existing else extra

        both_count = sum(1 for k in kept if k["matched_vector"] and k["matched_text"])
        logger.debug(
            f"chat retrieval (hybrid): appended {len(kept)} excerpts "
            f"({both_count} matched both signals, "
            f"top similarity={kept[0]['similarity']:.2f}, "
            f"top rrf={kept[0]['rrf_score']:.4f})"
        )

        # --- Retrieval preview (for UI) -----------------------------------
        # Light-weight record the streaming layer can ship to the frontend as
        # a single SSE event before any answer tokens flow. We deliberately
        # do NOT include chunk content here — the frontend just needs source
        # identity and a similarity hint to render a loading "Searching N
        # sources…" chip list. Full content stays in the `context` payload
        # the LLM sees.
        preview = [
            {
                "source_id": h["source_id"],
                "title": title_by_source.get(h["source_id"], "Untitled source"),
                "similarity": round(h["similarity"], 3),
                "matched_vector": h["matched_vector"],
                "matched_text": h["matched_text"],
            }
            for h in kept
        ]

        return {"context": merged, "retrieval_preview": preview}
    except Exception as e:
        # Never let retrieval failures block the chat. Log at debug so we
        # don't spam logs on fresh notebooks with no embeddings yet.
        logger.debug(f"chat retrieval node suppressed error: {e}")
        return {}


# --- Citation verification -------------------------------------------------
#
# The system prompt tells the LLM to cite with [source:id], [note:id], or
# [insight:id] tokens. Despite the explicit instructions and examples, LLMs
# still occasionally hallucinate IDs — either copying the example IDs from the
# prompt verbatim, or inventing plausible-looking random strings. A hallucinated
# citation is worse than no citation: it looks authoritative to the user, but
# the "Read more" affordance either 404s or (worse) lands on an unrelated
# record.
#
# This post-processing pass runs after every LLM response: extract every
# [table:id] token, batch-query the DB to see which ones actually exist, and
# strip the ones that don't. We deliberately strip rather than rewrite — the
# surrounding sentence is usually fine without the bracketed reference, and
# we'd rather over-remove than risk misattribution.
#
# Prefix → table mapping. Critical that this matches the system prompt's
# advertised prefixes. `insight` is the user-visible prefix; the underlying
# SurrealDB table is `source_insight`.
_CITATION_TABLE_BY_PREFIX = {
    "source": "source",
    "note": "note",
    "insight": "source_insight",
}

# Match [prefix:id] where id is any run of non-bracket, non-whitespace chars.
# SurrealDB RecordIDs after parse_record_ids() are "table:key" strings where
# the key is usually alphanumeric + underscore, but we stay liberal here and
# let the DB tell us whether the ID is real.
_CITATION_RE = re.compile(
    r"\[(source|note|insight):([^\]\s]+)\]"
)


async def _verify_and_strip_citations(content: str) -> tuple[str, int]:
    """Return ``(cleaned_content, num_stripped)``.

    Extracts every ``[table:id]`` citation, batches one SurrealQL query per
    table to check which IDs exist, and removes every citation token whose ID
    wasn't found. Any failure (DB hiccup, unexpected table, malformed ID)
    leaves the content untouched — we never want a citation-check regression
    to break chat.
    """
    if not content:
        return content, 0

    matches = list(_CITATION_RE.finditer(content))
    if not matches:
        return content, 0

    # Bucket the candidate IDs by their logical table. We store the full
    # `table:id` string (which is what SurrealDB's `id` field equals when
    # compared with a string literal).
    cited_by_prefix: dict[str, set[str]] = {}
    for m in matches:
        prefix = m.group(1)
        key = m.group(2)
        full_id = f"{_CITATION_TABLE_BY_PREFIX[prefix]}:{key}"
        cited_by_prefix.setdefault(prefix, set()).add(full_id)

    # One round-trip per table. SurrealDB's `type::thing` would be stricter
    # but `IN $ids` comparing record IDs against strings works because
    # parse_record_ids stringifies RecordIDs on the way out, and the
    # comparison side is a string literal list.
    valid_ids: set[str] = set()
    for prefix, ids in cited_by_prefix.items():
        table = _CITATION_TABLE_BY_PREFIX[prefix]
        try:
            rows = await repo_query(
                f"SELECT id FROM {table} WHERE id IN $ids",
                {"ids": list(ids)},
            )
        except Exception as e:
            # If verification fails, bail out entirely — don't silently strip
            # every citation just because the DB blipped.
            logger.debug(f"citation verify: query for {table} failed: {e}")
            return content, 0
        for row in rows or []:
            rid = row.get("id")
            if rid is None:
                continue
            valid_ids.add(str(rid))

    def _replace(m: re.Match) -> str:
        prefix = m.group(1)
        key = m.group(2)
        full_id = f"{_CITATION_TABLE_BY_PREFIX[prefix]}:{key}"
        if full_id in valid_ids:
            return m.group(0)
        return ""  # strip the hallucinated citation

    # Count how many citation tokens will be dropped BEFORE we mutate. This
    # counts occurrences (not unique IDs) so a hallucinated ID cited three
    # times registers as three strips.
    stripped = sum(
        1
        for m in matches
        if f"{_CITATION_TABLE_BY_PREFIX[m.group(1)]}:{m.group(2)}" not in valid_ids
    )

    cleaned = _CITATION_RE.sub(_replace, content)
    # Collapse the " ." / "  " artifacts that stripping can leave behind.
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned, stripped


async def call_model_with_messages(
    state: ThreadState, config: RunnableConfig
) -> dict:
    """Native async chat node.

    Runs on the FastAPI event loop — no ThreadPoolExecutor, no nested event
    loops, and the SurrealDB connection pool is reused across calls.
    """
    try:
        system_prompt = Prompter(prompt_template="chat/system").render(data=state)  # type: ignore[arg-type]
        payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        # Smart routing: if the caller hasn't pinned a model (no explicit
        # override) and both cheap/strong models are configured via env,
        # pick one based on question difficulty. Otherwise leave model_id
        # untouched — user UI choices and provision_langchain_model's own
        # fallback chain stay authoritative.
        if not model_id:
            # Find the most recent human turn — that's the question we
            # route on. Falls back to empty string if the message list is
            # somehow malformed; _route_model handles that defensively.
            last_human_text = ""
            for m in reversed(state.get("messages", []) or []):
                if isinstance(m, HumanMessage):
                    last_human_text = extract_text_content(m.content) or ""
                    break
            retrieval_preview = state.get("retrieval_preview") or []
            routed = _route_model(
                question=last_human_text,
                context_chars=len(system_prompt or ""),
                excerpt_count=len(retrieval_preview),
            )
            if routed:
                logger.debug(
                    f"model router → {routed} "
                    f"(q_chars={len(last_human_text)}, "
                    f"ctx_chars={len(system_prompt or '')}, "
                    f"excerpts={len(retrieval_preview)})"
                )
                model_id = routed

        model = await provision_langchain_model(
            str(payload), model_id, "chat", max_tokens=8192
        )

        ai_message = await model.ainvoke(payload)

        # Clean thinking content from AI response (e.g. <think>...</think> tags)
        content = extract_text_content(ai_message.content)
        cleaned_content = clean_thinking_content(content)

        # Verify every [table:id] citation against the DB and strip any
        # hallucinated ones. Best-effort: any failure returns content
        # unchanged rather than blowing up the chat turn.
        try:
            cleaned_content, stripped = await _verify_and_strip_citations(
                cleaned_content
            )
            if stripped:
                logger.debug(
                    f"citation verify: stripped {stripped} hallucinated citation(s)"
                )
        except Exception as e:
            logger.debug(f"citation verify: suppressed error: {e}")

        cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

        return {"messages": cleaned_message}
    except OpenNotebookError:
        raise
    except Exception as e:
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


# Uncompiled graph definition — safe to build at import time.
#
# Graph layout:
#   START → retrieve (augment context with semantic-search excerpts) → agent → END
#
# The retrieve step is intentionally a separate node (rather than inline in
# the agent node) so it's independently observable in LangSmith traces and
# can be toggled off without touching the LLM call site.
agent_state = StateGraph(ThreadState)
agent_state.add_node("retrieve", retrieve_from_embeddings)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_edge(START, "retrieve")
agent_state.add_edge("retrieve", "agent")
agent_state.add_edge("agent", END)


# Lazy async checkpointer + compiled graph.
#
# AsyncSqliteSaver.__init__ calls asyncio.get_running_loop() and builds an
# asyncio.Lock, so it must be constructed inside a running event loop. We
# therefore defer compilation until the first request and memoise the result.
# The sync `SqliteSaver` we used previously raises NotImplementedError on every
# async method (ainvoke, astream_events, aget_state), which broke SSE chat
# streaming.
_graph = None
_graph_lock: Optional[asyncio.Lock] = None


async def get_graph():
    """Return the compiled chat graph, building it on first call.

    The graph is wired up with an `AsyncSqliteSaver` checkpointer so that
    `ainvoke`, `astream_events`, and `aget_state` work natively on the event
    loop without offloading to threads.
    """
    global _graph, _graph_lock
    if _graph is not None:
        return _graph
    if _graph_lock is None:
        _graph_lock = asyncio.Lock()
    async with _graph_lock:
        if _graph is not None:
            return _graph
        conn = await aiosqlite.connect(LANGGRAPH_CHECKPOINT_FILE)
        memory = AsyncSqliteSaver(conn)
        await memory.setup()
        _graph = agent_state.compile(checkpointer=memory)
    return _graph
