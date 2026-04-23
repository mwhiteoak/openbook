import asyncio
from typing import Annotated, Dict, List, Optional

import aiosqlite
from ai_prompter import Prompter
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Source, SourceInsight
from open_notebook.exceptions import OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.context_builder import ContextBuilder
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content


class SourceChatState(TypedDict):
    messages: Annotated[list, add_messages]
    source_id: str
    # NOTE: do NOT persist Source / SourceInsight pydantic objects here —
    # AsyncSqliteSaver serializes checkpoints with msgpack, which rejects
    # arbitrary pydantic models. We rebuild them from source_id inside the
    # node, so they only need to live as locals.
    context: Optional[str]
    model_override: Optional[str]
    context_indicators: Optional[Dict[str, List[str]]]


async def call_model_with_source_context(
    state: SourceChatState, config: RunnableConfig
) -> dict:
    """Native async source-chat node.

    Runs on the FastAPI event loop — no ThreadPoolExecutor, no nested event
    loops. Builds source context once via ContextBuilder and reuses typed
    objects directly instead of re-hydrating them from the dict.
    """
    try:
        return await _call_model_with_source_context_inner(state, config)
    except OpenNotebookError:
        raise
    except Exception as e:
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


async def _call_model_with_source_context_inner(
    state: SourceChatState, config: RunnableConfig
) -> dict:
    source_id = state.get("source_id")
    if not source_id:
        raise ValueError("source_id is required in state")

    # Build source context using ContextBuilder — directly awaitable, no loop
    # gymnastics, and uses the shared SurrealDB connection pool.
    context_builder = ContextBuilder(
        source_id=source_id,
        include_insights=True,
        include_notes=False,  # Focus on source-specific content
        # Source chat needs the raw document text in context so the LLM can
        # answer content-specific questions (e.g. "how much is the quote?").
        # Default "insights" level omits full_text — leaving the model with
        # just the title. "full content" pulls full_text into the payload.
        source_inclusion_level="full content",
        max_tokens=50000,  # Reasonable limit for source context
    )
    context_data = await context_builder.build()

    # Re-use the typed ContextItem objects the builder already constructed
    # instead of hydrating fresh Source/SourceInsight pydantic models from the
    # dict payload a second time.
    source: Optional[Source] = None
    insights: List[SourceInsight] = []
    context_indicators: Dict[str, List[str]] = {
        "sources": [],
        "insights": [],
        "notes": [],
    }

    for item in context_builder.items:
        if item.type == "source":
            # Items store the raw context dict; we only need the id for
            # indicators, and `get_source_reference()` uses `source.id` only
            # through domain calls the template is not making.  Keep a minimal
            # reference so the template can still call `.model_dump()`.
            if item.id:
                context_indicators["sources"].append(item.id)
            # Lazy-load the actual domain object only if prompt needs it
            if source is None and item.id:
                try:
                    source = await Source.get(item.id)
                except Exception:
                    source = None
        elif item.type == "insight":
            if item.id:
                context_indicators["insights"].append(item.id)
            content = item.content or {}
            try:
                insights.append(
                    SourceInsight(
                        id=content.get("id"),
                        insight_type=content.get("insight_type"),
                        content=content.get("content"),
                    )
                )
            except Exception:
                # Fall back to raw dict-based attribute access if SourceInsight
                # validation fails for any reason.
                continue

    # Format context for the prompt
    formatted_context = _format_source_context(context_data)

    # Build prompt data for the template
    prompt_data = {
        "source": source.model_dump() if source else None,
        "insights": [insight.model_dump() for insight in insights] if insights else [],
        "context": formatted_context,
        "context_indicators": context_indicators,
    }

    # Apply the source_chat prompt template
    system_prompt = Prompter(prompt_template="source_chat/system").render(
        data=prompt_data
    )
    payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    model_id = config.get("configurable", {}).get("model_id") or state.get(
        "model_override"
    )
    model = await provision_langchain_model(
        str(payload), model_id, "chat", max_tokens=8192
    )

    ai_message = await model.ainvoke(payload)

    # Clean thinking content from AI response (e.g. <think>...</think> tags)
    content = extract_text_content(ai_message.content)
    cleaned_content = clean_thinking_content(content)
    cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

    # Intentionally omit `source` / `insights` — they're pydantic objects and
    # AsyncSqliteSaver's msgpack serializer can't encode them. They're purely
    # local to this node (rebuilt from source_id every call), so dropping them
    # from the returned state dict prevents checkpoint-write failures.
    return {
        "messages": cleaned_message,
        "context": formatted_context,
        "context_indicators": context_indicators,
    }


def _format_source_context(context_data: Dict) -> str:
    """
    Format the context data into a readable string for the prompt.

    Args:
        context_data: Context data from ContextBuilder

    Returns:
        Formatted context string
    """
    context_parts = []

    # Add source information
    if context_data.get("sources"):
        context_parts.append("## SOURCE CONTENT")
        for source in context_data["sources"]:
            if isinstance(source, dict):
                context_parts.append(f"**Source ID:** {source.get('id', 'Unknown')}")
                context_parts.append(f"**Title:** {source.get('title', 'No title')}")
                if source.get("full_text"):
                    # Truncate full text if too long
                    full_text = source["full_text"]
                    if len(full_text) > 5000:
                        full_text = full_text[:5000] + "...\n[Content truncated]"
                    context_parts.append(f"**Content:**\n{full_text}")
                context_parts.append("")  # Empty line for separation

    # Add insights
    if context_data.get("insights"):
        context_parts.append("## SOURCE INSIGHTS")
        for insight in context_data["insights"]:
            if isinstance(insight, dict):
                context_parts.append(f"**Insight ID:** {insight.get('id', 'Unknown')}")
                context_parts.append(
                    f"**Type:** {insight.get('insight_type', 'Unknown')}"
                )
                context_parts.append(
                    f"**Content:** {insight.get('content', 'No content')}"
                )
                context_parts.append("")  # Empty line for separation

    # Add metadata
    if context_data.get("metadata"):
        metadata = context_data["metadata"]
        context_parts.append("## CONTEXT METADATA")
        context_parts.append(f"- Source count: {metadata.get('source_count', 0)}")
        context_parts.append(f"- Insight count: {metadata.get('insight_count', 0)}")
        context_parts.append(f"- Total tokens: {context_data.get('total_tokens', 0)}")
        context_parts.append("")

    return "\n".join(context_parts)


# Uncompiled graph definition — safe at import time.
source_chat_state = StateGraph(SourceChatState)
source_chat_state.add_node("source_chat_agent", call_model_with_source_context)
source_chat_state.add_edge(START, "source_chat_agent")
source_chat_state.add_edge("source_chat_agent", END)


# Lazy async checkpointer + compiled graph.
#
# See open_notebook/graphs/chat.py for the full rationale — sync `SqliteSaver`
# raises NotImplementedError on every async method, which broke `ainvoke` and
# `astream_events` for source chat. We use `AsyncSqliteSaver` which must be
# constructed inside a running event loop, hence the lazy factory.
_graph = None
_graph_lock: Optional[asyncio.Lock] = None


async def get_source_chat_graph():
    """Return the compiled source-chat graph, building it on first call."""
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
        _graph = source_chat_state.compile(checkpointer=memory)
    return _graph
