"""
Fingerprinting for the chat Q&A cache.

The cache decides whether two questions can share an answer by comparing a
**context fingerprint**: a stable hash of everything that would change what
the LLM sees. If any input drifts (a source was edited, the model changed,
the user toggled a different inclusion level) the fingerprint differs and
the cache entry is effectively isolated from the new request.

What goes into the fingerprint:
  * Scope (notebook_id or source_id — whichever the chat surface is bound to)
  * Selected context: {source_id: inclusion_level, ...} and {note_id: level, ...}
    Order-independent — we sort keys.
  * Model id — different models produce materially different answers; we
    don't want to serve a cheap-model answer to a premium-model request.
  * max(source.updated) across the selected sources — flips the fingerprint
    the moment any underlying content is re-ingested or edited. This is
    the "auto-invalidate on source change" behaviour the user asked for.

Output: a short hex digest (blake2b, 16 bytes → 32 hex chars). Collisions
for cache lookup are acceptable at this width because we also match on
normalized question text and scope — the fingerprint only has to
discriminate between *different* context configurations, not be globally
unique.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional


def normalize_question(question: str) -> str:
    """Lowercase + collapse whitespace for exact-match comparison.

    Users routinely phrase the same question with different casing or stray
    spaces ("What is the rent?" vs "what is the rent?  "). Normalising both
    the stored value and the lookup key lets us catch these as exact hits
    before falling back to the more expensive semantic search.
    """
    if not question:
        return ""
    # Strip, lowercase, collapse any run of whitespace (including newlines).
    return re.sub(r"\s+", " ", question.strip().lower())


def _stable_dict(d: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return dict with string keys sorted for stable JSON serialization.

    Selection maps come from the frontend in arbitrary iteration order; we
    sort to guarantee identical selections produce identical hashes.
    """
    if not d:
        return {}
    return {k: d[k] for k in sorted(d.keys())}


def _coerce_iso(value: Any) -> Optional[str]:
    """Best-effort conversion of timestamps to ISO 8601 strings.

    SurrealDB returns datetimes as native `datetime` objects via the async
    driver but as ISO strings in some query paths. Normalise to string so
    max() comparisons and hashing are consistent.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def compute_context_fingerprint(
    *,
    notebook_id: Optional[str] = None,
    source_id: Optional[str] = None,
    context_config: Optional[Mapping[str, Any]] = None,
    model_id: Optional[str] = None,
    source_updated_timestamps: Optional[Iterable[Any]] = None,
) -> str:
    """Compute a deterministic fingerprint for the given chat context.

    Args:
        notebook_id: Scope identifier for notebook chats (mutually exclusive
            with source_id).
        source_id: Scope identifier for source-level chats.
        context_config: The ``{"sources": {...}, "notes": {...}}`` mapping
            passed to the chat graph — exactly what the LLM's context builder
            sees. Unknown keys are included in the hash verbatim so future
            config additions don't silently collide.
        model_id: Chat model identifier (record id or esperanto name).
        source_updated_timestamps: Iterable of ``updated`` timestamps from
            the selected sources. We fold max() into the hash so any edit
            bumps the fingerprint without us having to track individual
            source state.

    Returns:
        32-char hex digest (blake2b, 16 bytes).
    """
    # Pull out the two selection maps (if any) and stabilise their ordering.
    cfg = context_config or {}
    sources = _stable_dict(cfg.get("sources") if isinstance(cfg, Mapping) else None)
    notes = _stable_dict(cfg.get("notes") if isinstance(cfg, Mapping) else None)

    # Everything else from the config that we didn't destructure — keep it in
    # the hash so e.g. include_insights toggles still affect cache identity.
    other_cfg = {
        k: v
        for k, v in (cfg.items() if isinstance(cfg, Mapping) else [])
        if k not in ("sources", "notes")
    }
    other_cfg = _stable_dict(other_cfg)

    # max() of updated timestamps — guards against stale cache when content
    # underlying the answer has been edited.
    max_updated: Optional[str] = None
    if source_updated_timestamps:
        iso_values = [v for v in (_coerce_iso(x) for x in source_updated_timestamps) if v]
        if iso_values:
            max_updated = max(iso_values)

    payload = {
        "notebook_id": notebook_id,
        "source_id": source_id,
        "sources": sources,
        "notes": notes,
        "other_cfg": other_cfg,
        "model_id": model_id,
        "max_source_updated": max_updated,
    }

    # sort_keys for belt-and-braces stability (we already sorted internals,
    # but top-level insertion order shouldn't matter either).
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.blake2b(serialized.encode("utf-8"), digest_size=16).hexdigest()
