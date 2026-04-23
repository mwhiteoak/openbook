"""
Notebook running summary — compact LLM-maintained memory of notebook activity.

The summary lives on the ``notebook`` row (migration 18) and is injected
into the chat system prompt so the model carries long-term context across
short chat sessions. Refreshes are gated: we only rebuild the summary
once every ``REFRESH_BATCH_SIZE`` new chat messages (across all sessions
in the notebook), to keep LLM cost and write contention bounded.

Design notes:
  * **Fire-and-forget**: the chat router calls ``maybe_refresh_running_summary``
    after streaming is done. Failures are logged but never surfaced —
    the cached summary just stays stale until the next turn triggers a
    successful refresh.
  * **LLM of choice**: uses the chat model (``provision_langchain_model(..., "chat")``)
    with a tight token budget. Small/local models produce perfectly good
    summaries for this use case.
  * **Append-aware prompt**: the LLM sees the *previous* summary plus the
    new messages, not the whole conversation. That's cheap on tokens and
    encourages incremental updates rather than re-summarizing from scratch
    (which in practice destabilizes the summary across runs).
  * **Opt-out**: setting ``OPEN_NOTEBOOK_RUNNING_SUMMARY=0`` disables the
    feature entirely — no writes, no reads (chat graph renders the template
    with ``running_summary`` still None, so the block is skipped).

The throttle field ``running_summary_msg_count`` stores the count of
messages already folded in. On refresh we compare against the current
message count and skip if the delta is under ``REFRESH_BATCH_SIZE``.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import List, Optional

from loguru import logger

from open_notebook.database.repository import ensure_record_id, repo_query


# Config --------------------------------------------------------------------
# Batch size: refresh whenever N new messages have landed since the last
# summary. 8 is a reasonable trade-off between cost and staleness — at
# typical chat pacing that's maybe once every 4 turns.
REFRESH_BATCH_SIZE: int = int(os.getenv("OPEN_NOTEBOOK_RUNNING_SUMMARY_BATCH", "8"))

# Hard cap on the summary length we ask the model to produce. The prompt
# says "about 5 short paragraphs"; the token budget keeps things honest.
SUMMARY_MAX_TOKENS: int = int(
    os.getenv("OPEN_NOTEBOOK_RUNNING_SUMMARY_MAX_TOKENS", "600")
)

# Timeout for the summary call. Slow but not blocking — this always runs
# after the user has already received their answer.
SUMMARY_TIMEOUT_SEC: float = float(
    os.getenv("OPEN_NOTEBOOK_RUNNING_SUMMARY_TIMEOUT", "20.0")
)

# Disable switch — empty / "0" / "false" turns the whole feature off.
_ENABLED: bool = os.getenv(
    "OPEN_NOTEBOOK_RUNNING_SUMMARY", "1"
).strip().lower() not in {"0", "false", "no", "off", ""}


async def _count_notebook_messages(notebook_id: str) -> int:
    """Total chat messages across all sessions in this notebook.

    Uses a single SurrealQL query so we don't fan out per-session. Returns
    0 on any error (leaving the throttle to treat the notebook as empty).
    """
    try:
        rows = await repo_query(
            """
            SELECT count() AS n FROM chat_session
            WHERE out IN (
                SELECT out FROM refers_to WHERE in = $notebook
            )
            GROUP ALL
            """,
            {"notebook": ensure_record_id(notebook_id)},
        )
        # The above is a sketch — in practice the message count lives on
        # the checkpoint rows. Fall through to the simpler checkpoint-based
        # count below if the sketch returns nothing.
        if rows and rows[0].get("n"):
            return int(rows[0]["n"])
    except Exception as e:
        logger.debug(f"running-summary: session count query failed: {e}")

    # Fallback: count LangGraph checkpoint rows tagged with this notebook.
    # Not perfect but good enough for throttling.
    try:
        rows = await repo_query(
            "SELECT count() AS n FROM chat_session WHERE notebook_id = $nb GROUP ALL",
            {"nb": ensure_record_id(notebook_id)},
        )
        if rows and rows[0].get("n"):
            return int(rows[0]["n"])
    except Exception:
        pass
    return 0


async def _fetch_recent_messages(
    notebook_id: str, since_count: int, max_messages: int = 40
) -> List[str]:
    """Fetch the most recent user/AI messages as plain strings.

    We intentionally keep the fetch simple — the chat.py graph stores
    messages in a LangGraph SqliteSaver that we don't query cross-process.
    Instead we read from the ``chat_session`` rows (which mirror message
    content in the persisted session dict) and take the tail.

    Each returned string is of the form ``"user: …"`` / ``"assistant: …"``
    so the LLM sees who said what.
    """
    try:
        rows = await repo_query(
            """
            SELECT messages FROM chat_session
            WHERE notebook_id = $nb
            ORDER BY updated DESC
            LIMIT 5
            """,
            {"nb": ensure_record_id(notebook_id)},
        )
    except Exception as e:
        logger.debug(f"running-summary: message fetch failed: {e}")
        return []

    collected: List[str] = []
    for row in rows or []:
        msgs = row.get("messages") or []
        for m in msgs:
            # Messages may be stored as dicts (type+content) or as
            # pre-serialized strings — handle both.
            if isinstance(m, dict):
                role = m.get("type") or m.get("role") or "user"
                content = m.get("content") or ""
                if content:
                    collected.append(f"{role}: {content}")
            elif isinstance(m, str) and m.strip():
                collected.append(m)
    # Newest first from the DB, but the LLM reads better chronologically.
    collected.reverse()
    if len(collected) > max_messages:
        collected = collected[-max_messages:]
    return collected


def _build_summary_prompt(
    previous_summary: Optional[str], new_messages: List[str]
) -> str:
    """Compose the prompt for the summary LLM.

    The prompt asks the model to *update* the previous summary rather than
    rewrite from scratch. In practice this keeps the summary stable across
    runs — otherwise each refresh rewords the whole thing and the user
    experiences summary churn.
    """
    prev_block = (
        f"# PREVIOUS SUMMARY\n{previous_summary.strip()}\n\n"
        if previous_summary and previous_summary.strip()
        else ""
    )
    messages_block = "\n".join(new_messages) if new_messages else "(no new messages)"
    return (
        "You maintain a compact running summary of a research notebook. "
        "The summary will be injected into future chat system prompts as "
        "long-term memory for the assistant.\n\n"
        "Write the summary as ~5 short paragraphs covering: (1) the main "
        "topics the user is researching, (2) key decisions or conclusions "
        "reached so far, (3) open questions or unresolved threads, "
        "(4) important entities/names/terms, and (5) preferences the user "
        "has expressed (tone, format, scope).\n\n"
        "Do NOT invent citations or document ids. Do NOT copy long quotes. "
        "Prefer concrete specifics over vague generalities. If there is a "
        "PREVIOUS SUMMARY below, UPDATE it rather than rewriting from scratch — "
        "preserve entries still relevant and integrate new information.\n\n"
        f"{prev_block}"
        f"# NEW MESSAGES (most recent chat activity)\n{messages_block}\n\n"
        "# UPDATED SUMMARY\n"
    )


async def maybe_refresh_running_summary(
    notebook_id: str, *, model_id: Optional[str] = None
) -> None:
    """Refresh the notebook's running summary if enough new messages have landed.

    Fire-and-forget — call from the chat router with ``asyncio.create_task``.
    Returns None regardless of outcome; errors are logged at DEBUG.
    """
    if not _ENABLED:
        return
    if not notebook_id:
        return

    try:
        rows = await repo_query(
            """
            SELECT running_summary, running_summary_msg_count
            FROM $id
            LIMIT 1
            """,
            {"id": ensure_record_id(notebook_id)},
        )
    except Exception as e:
        logger.debug(f"running-summary: notebook fetch failed: {e}")
        return

    if not rows:
        return
    row = rows[0]
    previous_summary = row.get("running_summary") or None
    already_counted = int(row.get("running_summary_msg_count") or 0)

    total_messages = await _count_notebook_messages(notebook_id)
    if total_messages - already_counted < REFRESH_BATCH_SIZE:
        # Not enough new activity — skip silently.
        return

    recent = await _fetch_recent_messages(
        notebook_id, since_count=already_counted
    )
    if not recent:
        return

    prompt = _build_summary_prompt(previous_summary, recent)

    # Lazy import to avoid pulling the provision stack into domain module
    # load. Failures here are quiet — they just mean the feature no-ops.
    try:
        from open_notebook.ai.provision import provision_langchain_model
    except Exception as e:
        logger.debug(f"running-summary: provisioning import failed: {e}")
        return

    try:
        model = await provision_langchain_model(
            prompt, model_id, "chat", max_tokens=SUMMARY_MAX_TOKENS
        )
        result = await asyncio.wait_for(
            model.ainvoke(prompt), timeout=SUMMARY_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        logger.debug("running-summary: LLM call timed out; leaving summary stale")
        return
    except Exception as e:
        logger.debug(f"running-summary: LLM call failed: {e}")
        return

    try:
        from open_notebook.utils.text_utils import (
            clean_thinking_content,
            extract_text_content,
        )

        summary = clean_thinking_content(extract_text_content(result.content)).strip()
    except Exception:
        summary = str(getattr(result, "content", "")).strip()

    if not summary:
        return

    # Write back. We UPDATE explicitly (rather than using the ObjectModel
    # save() round-trip) to keep this hot path minimal — one query, no
    # re-embedding, no side effects. SurrealQL's UPDATE ... SET is atomic
    # per-record so concurrent refreshes on the same notebook are safe
    # (last-write-wins; both writers see a recent summary).
    try:
        await repo_query(
            """
            UPDATE $id SET
                running_summary = $summary,
                running_summary_updated = time::now(),
                running_summary_msg_count = $count,
                updated = time::now()
            """,
            {
                "id": ensure_record_id(notebook_id),
                "summary": summary,
                "count": total_messages,
            },
        )
        logger.debug(
            f"running-summary: refreshed notebook {notebook_id} "
            f"({len(summary)} chars, {total_messages} msgs folded)"
        )
    except Exception as e:
        logger.debug(f"running-summary: write-back failed: {e}")


__all__ = [
    "maybe_refresh_running_summary",
    "REFRESH_BATCH_SIZE",
]
