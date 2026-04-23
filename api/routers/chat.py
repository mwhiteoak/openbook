import asyncio
import json
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from api.auth import get_current_user

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.chat_cache import (
    bump_cache_hit,
    find_cached_answer,
    save_cached_answer,
)
from open_notebook.domain.notebook import ChatSession, Note, Notebook, Source
from open_notebook.exceptions import (
    NotFoundError,
)
from open_notebook.graphs.chat import get_graph as get_chat_graph
from open_notebook.utils.cache_fingerprint import compute_context_fingerprint
from open_notebook.utils.graph_utils import get_session_message_count

router = APIRouter()


# Request/Response models
class CreateSessionRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID to create session for")
    title: Optional[str] = Field(None, description="Optional session title")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this session"
    )


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatMessage(BaseModel):
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (human|ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatSessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    notebook_id: Optional[str] = Field(None, description="Notebook ID")
    created: str = Field(..., description="Creation timestamp")
    updated: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(
        None, description="Number of messages in session"
    )
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatSessionWithMessagesResponse(ChatSessionResponse):
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Session messages"
    )


class ExecuteChatRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message content")
    context: Dict[str, Any] = Field(
        ..., description="Chat context with sources and notes"
    )
    model_override: Optional[str] = Field(
        None, description="Optional model override for this message"
    )
    # Frontend uses this flag when the user clicks "Regenerate" on a cached
    # answer — we skip the cache lookup but still WRITE the fresh answer to
    # the cache, which transparently replaces the old entry the next time a
    # matching question is asked.
    bypass_cache: bool = Field(
        False, description="Skip Q&A cache lookup (e.g. regenerate click)"
    )
    # Scope hint for the Q&A cache. The chat graph itself doesn't need this,
    # but the cache key does — and the session<->notebook link costs an extra
    # DB round-trip to resolve server-side, so the frontend passes it through.
    notebook_id: Optional[str] = Field(
        None, description="Notebook scope for Q&A cache (optional)"
    )


class ExecuteChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Updated message list")


class BuildContextRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID")
    context_config: Dict[str, Any] = Field(..., description="Context configuration")


class BuildContextResponse(BaseModel):
    context: Dict[str, Any] = Field(..., description="Built context data")
    token_count: int = Field(..., description="Estimated token count")
    char_count: int = Field(..., description="Character count")


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")


@router.get("/chat/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(notebook_id: str = Query(..., description="Notebook ID")):
    """Get all chat sessions for a notebook."""
    try:
        # Get notebook to verify it exists
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Get sessions for this notebook
        sessions_list = await notebook.get_chat_sessions()

        chat_graph = await get_chat_graph()
        results = []
        for session in sessions_list:
            session_id = str(session.id)

            # Get message count from LangGraph state
            msg_count = await get_session_message_count(chat_graph, session_id)

            results.append(
                ChatSessionResponse(
                    id=session.id or "",
                    title=session.title or "Untitled Session",
                    notebook_id=notebook_id,
                    created=str(session.created),
                    updated=str(session.updated),
                    message_count=msg_count,
                    model_override=getattr(session, "model_override", None),
                )
            )

        return results
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching chat sessions: {str(e)}"
        )


@router.post("/chat/sessions", response_model=ChatSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Create a new chat session."""
    try:
        from open_notebook.database.repository import ensure_record_id

        # Verify notebook exists
        notebook = await Notebook.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Create new session
        session = ChatSession(
            title=request.title
            or f"Chat Session {asyncio.get_event_loop().time():.0f}",
            model_override=request.model_override,
            user_id=str(ensure_record_id(current_user["id"])),
        )
        await session.save()

        # Relate session to notebook
        await session.relate_to_notebook(request.notebook_id)

        return ChatSessionResponse(
            id=session.id or "",
            title=session.title or "",
            notebook_id=request.notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=0,
            model_override=session.model_override,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating chat session: {str(e)}"
        )


@router.get(
    "/chat/sessions/{session_id}", response_model=ChatSessionWithMessagesResponse
)
async def get_session(session_id: str):
    """Get a specific session with its messages."""
    try:
        # Get session
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get session state from LangGraph to retrieve messages (async checkpointer).
        chat_graph = await get_chat_graph()
        thread_state = await chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )

        # Extract messages from state
        messages: list[ChatMessage] = []
        if thread_state and thread_state.values and "messages" in thread_state.values:
            for msg in thread_state.values["messages"]:
                messages.append(
                    ChatMessage(
                        id=getattr(msg, "id", f"msg_{len(messages)}"),
                        type=msg.type if hasattr(msg, "type") else "unknown",
                        content=msg.content if hasattr(msg, "content") else str(msg),
                        timestamp=None,  # LangChain messages don't have timestamps by default
                    )
                )

        # Find notebook_id (we need to query the relationship)
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )

        notebook_query = await repo_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )

        notebook_id = notebook_query[0]["out"] if notebook_query else None

        if not notebook_id:
            # This might be an old session created before API migration
            logger.warning(
                f"No notebook relationship found for session {session_id} - may be an orphaned session"
            )

        return ChatSessionWithMessagesResponse(
            id=session.id or "",
            title=session.title or "Untitled Session",
            notebook_id=notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=len(messages),
            messages=messages,
            model_override=getattr(session, "model_override", None),
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error fetching session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching session: {str(e)}")


@router.put("/chat/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update session title."""
    try:
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        update_data = request.model_dump(exclude_unset=True)

        if "title" in update_data:
            session.title = update_data["title"]

        if "model_override" in update_data:
            session.model_override = update_data["model_override"]

        await session.save()

        # Find notebook_id
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        notebook_query = await repo_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )
        notebook_id = notebook_query[0]["out"] if notebook_query else None

        # Get message count from LangGraph state
        chat_graph = await get_chat_graph()
        msg_count = await get_session_message_count(chat_graph, full_session_id)

        return ChatSessionResponse(
            id=session.id or "",
            title=session.title or "",
            notebook_id=notebook_id,
            created=str(session.created),
            updated=str(session.updated),
            message_count=msg_count,
            model_override=session.model_override,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")


@router.delete("/chat/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(session_id: str):
    """Delete a chat session."""
    try:
        # Ensure session_id has proper table prefix
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        await session.delete()

        return SuccessResponse(success=True, message="Session deleted successfully")
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@router.post("/chat/execute", response_model=ExecuteChatResponse)
async def execute_chat(request: ExecuteChatRequest):
    """Execute a chat request and get AI response."""
    try:
        # Verify session exists
        # Ensure session_id has proper table prefix
        full_session_id = (
            request.session_id
            if request.session_id.startswith("chat_session:")
            else f"chat_session:{request.session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Determine model override (per-request override takes precedence over session-level)
        model_override = (
            request.model_override
            if request.model_override is not None
            else getattr(session, "model_override", None)
        )

        # Get current state (async checkpointer supports aget_state natively).
        chat_graph = await get_chat_graph()
        current_state = await chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )

        # Prepare state for execution
        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        state_values["context"] = request.context
        state_values["model_override"] = model_override

        # Add user message to state
        user_message = HumanMessage(content=request.message)
        state_values["messages"].append(user_message)

        # Execute chat graph natively — the node is now async, so we run on
        # the FastAPI event loop directly (no thread hop, no re-authentication
        # of the SurrealDB pool, no nested event loops).
        result = await chat_graph.ainvoke(
            input=state_values,  # type: ignore[arg-type]
            config=RunnableConfig(
                configurable={
                    "thread_id": full_session_id,
                    "model_id": model_override,
                }
            ),
        )

        # Update session timestamp
        await session.save()

        # Convert messages to response format
        messages: list[ChatMessage] = []
        for msg in result.get("messages", []):
            messages.append(
                ChatMessage(
                    id=getattr(msg, "id", f"msg_{len(messages)}"),
                    type=msg.type if hasattr(msg, "type") else "unknown",
                    content=msg.content if hasattr(msg, "content") else str(msg),
                    timestamp=None,
                )
            )

        return ExecuteChatResponse(session_id=request.session_id, messages=messages)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        # Log detailed error with context for debugging
        logger.error(
            f"Error executing chat: {str(e)}\n"
            f"  Session ID: {request.session_id}\n"
            f"  Model override: {request.model_override}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"Error executing chat: {str(e)}")


async def _compute_chat_fingerprint(
    *,
    notebook_id: Optional[str],
    source_id: Optional[str],
    context_config: Optional[Dict[str, Any]],
    model_override: Optional[str],
) -> str:
    """Build the Q&A cache fingerprint for a chat request.

    We collect ``updated`` timestamps for every source listed in the config
    (ignoring those marked ``not in``) so any source edit flips the
    fingerprint and transparently invalidates stale cache rows. Unknown
    sources are ignored — they simply don't contribute a timestamp.
    """
    timestamps: List[Any] = []
    cfg_sources = (context_config or {}).get("sources") or {}
    for sid, status in cfg_sources.items():
        if isinstance(status, str) and "not in" in status:
            continue
        full_id = sid if sid.startswith("source:") else f"source:{sid}"
        try:
            rows = await repo_query(
                "SELECT updated FROM $id",
                {"id": ensure_record_id(full_id)},
            )
            if rows and rows[0].get("updated"):
                timestamps.append(rows[0]["updated"])
        except Exception:
            # Missing source or query failure → drop it from the fingerprint;
            # cache will just be slightly more willing to hit.
            continue

    return compute_context_fingerprint(
        notebook_id=notebook_id,
        source_id=source_id,
        context_config=context_config,
        model_id=model_override,
        source_updated_timestamps=timestamps,
    )


async def _stream_cached_chat_answer(
    *,
    session_id: str,
    message: str,
    cache_row: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Replay a cached answer over SSE and update the LangGraph checkpoint.

    We still persist the Q&A pair into the session so the conversation
    history rendered from the checkpoint includes the cached turn (the
    frontend refetches messages after stream completion). The ``cached``
    flag tells the UI to render the "⚡ Cached · Regenerate" affordance.
    """
    answer_text = str(cache_row.get("answer") or "")
    cache_id = str(cache_row.get("id") or "")

    # Echo the user message so the UI's handler path matches the live case.
    yield "data: " + json.dumps(
        {"type": "user_message", "content": message, "timestamp": None}
    ) + "\n\n"

    # Emit a single ai_message with cached=True. We skip delta streaming —
    # there's nothing to tokenise and instant appearance is the whole point.
    ai_id = f"cached_{cache_id}" if cache_id else "cached"
    yield "data: " + json.dumps(
        {
            "type": "ai_message",
            "id": ai_id,
            "content": answer_text,
            "timestamp": None,
            "cached": True,
            "cache_id": cache_id,
        }
    ) + "\n\n"

    # Persist into LangGraph's checkpoint so the conversation history stays
    # coherent when the user reloads. We use aupdate_state with as_node="agent"
    # so the reducer (add_messages) appends both the user turn and the AI
    # reply. Failures here are non-fatal — the SSE response has already been
    # sent to the user.
    try:
        chat_graph = await get_chat_graph()
        await chat_graph.aupdate_state(
            config=RunnableConfig(configurable={"thread_id": session_id}),
            values={
                "messages": [
                    HumanMessage(content=message),
                    AIMessage(content=answer_text, id=ai_id),
                ]
            },
            as_node="agent",
        )
    except Exception as e:
        logger.warning(f"Failed to persist cached turn to checkpoint: {e}")

    # Bump the hit counter (fire-and-forget, doesn't affect SSE ordering).
    if cache_id:
        await bump_cache_hit(cache_id)

    yield "data: " + json.dumps({"type": "complete"}) + "\n\n"


async def stream_chat_response(
    session_id: str,
    message: str,
    context: Dict[str, Any],
    model_override: Optional[str] = None,
    *,
    notebook_id: Optional[str] = None,
    bypass_cache: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream the chat response as Server-Sent Events.

    Uses LangGraph's `astream_events` to forward AI token deltas to the
    frontend in real time.  Falls back to a single `ai_message` event if the
    model provider does not support token-level streaming.

    Before invoking the graph, consults the Q&A cache. On a hit we replay
    the stored answer (with `cached: true`) and persist the turn to the
    LangGraph checkpoint so session history stays consistent. On a miss we
    run the graph normally and write the final answer to the cache.
    """
    # ---- Cache lookup (notebook scope) ----------------------------------
    cache_fingerprint: Optional[str] = None
    if notebook_id:
        try:
            cache_fingerprint = await _compute_chat_fingerprint(
                notebook_id=notebook_id,
                source_id=None,
                context_config=context,
                model_override=model_override,
            )
        except Exception as e:
            logger.debug(f"Fingerprint computation failed, skipping cache: {e}")
            cache_fingerprint = None

        if cache_fingerprint and not bypass_cache:
            try:
                cache_hit = await find_cached_answer(
                    question=message,
                    context_fingerprint=cache_fingerprint,
                    notebook_id=notebook_id,
                    model_id=model_override,
                )
            except Exception as e:
                logger.warning(f"Cache lookup raised: {e}")
                cache_hit = None

            if cache_hit:
                async for chunk in _stream_cached_chat_answer(
                    session_id=session_id,
                    message=message,
                    cache_row=cache_hit,
                ):
                    yield chunk
                return

    try:
        # Async checkpointer supports aget_state natively.
        chat_graph = await get_chat_graph()
        current_state = await chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": session_id}),
        )

        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        state_values["context"] = context
        state_values["model_override"] = model_override

        user_message = HumanMessage(content=message)
        state_values["messages"].append(user_message)

        # Emit user message echo first
        yield "data: " + json.dumps(
            {"type": "user_message", "content": message, "timestamp": None}
        ) + "\n\n"

        # Stream token deltas via astream_events
        accumulated = ""
        ai_message_id: Optional[str] = None
        saw_chunk = False

        config = RunnableConfig(
            configurable={"thread_id": session_id, "model_id": model_override}
        )

        try:
            async for event in chat_graph.astream_events(
                input=state_values,  # type: ignore[arg-type]
                config=config,
                version="v2",
            ):
                ev_type = event.get("event")

                # Perplexity-style retrieval preview. The `retrieve` node
                # exposes a `retrieval_preview` list describing the source
                # passages that will feed the LLM. Forward this once, as soon
                # as retrieval finishes — the frontend can render "Searching
                # 3 sources…" chips before the answer tokens start flowing.
                if (
                    ev_type == "on_chain_end"
                    and event.get("name") == "retrieve"
                ):
                    output = (event.get("data") or {}).get("output") or {}
                    preview = output.get("retrieval_preview")
                    if preview:
                        yield "data: " + json.dumps(
                            {
                                "type": "retrieval_sources",
                                "sources": preview,
                            }
                        ) + "\n\n"
                    continue

                if ev_type == "on_chat_model_stream":
                    # CRITICAL: the retrieve node now makes its own LLM calls
                    # (follow-up rewrite, cross-encoder-style rerank). Those
                    # produce `on_chat_model_stream` events too — without this
                    # filter they'd leak into the user's answer as garbage
                    # tokens before the real response, and the UI would hang
                    # waiting to reconcile. Gate on langgraph_node so only the
                    # agent node's tokens reach the client.
                    node = (event.get("metadata") or {}).get("langgraph_node")
                    if node and node != "agent":
                        continue
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    # LangChain streaming chunks have a .content attr (str)
                    delta = getattr(chunk, "content", None)
                    if isinstance(delta, list):
                        # Some providers emit list-of-parts; join text parts
                        delta = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in delta
                        )
                    if not delta:
                        continue
                    saw_chunk = True
                    accumulated += delta
                    if ai_message_id is None:
                        ai_message_id = getattr(chunk, "id", None) or "ai_stream"
                    yield "data: " + json.dumps(
                        {
                            "type": "ai_message_delta",
                            "id": ai_message_id,
                            "delta": delta,
                        }
                    ) + "\n\n"
        except Exception:
            # Token streaming failed — fall back to non-streaming ainvoke below
            saw_chunk = False
            accumulated = ""

        final_answer_text: str = ""

        if not saw_chunk:
            # Fallback: run ainvoke and emit a single ai_message event
            result = await chat_graph.ainvoke(
                input=state_values,  # type: ignore[arg-type]
                config=config,
            )
            if "messages" in result:
                # We only want the NEWEST AI message (there will typically be
                # one per turn, but if the graph somehow produced several we
                # cache the final one — that's what the user sees as the
                # answer to this turn).
                for msg in result["messages"]:
                    if hasattr(msg, "type") and msg.type == "ai":
                        content_str = (
                            msg.content if hasattr(msg, "content") else str(msg)
                        )
                        final_answer_text = content_str
                        yield "data: " + json.dumps(
                            {
                                "type": "ai_message",
                                "id": getattr(msg, "id", "ai_msg"),
                                "content": content_str,
                                "timestamp": None,
                            }
                        ) + "\n\n"
        else:
            # Streaming completed; emit a final ai_message with the accumulated
            # content so clients that missed deltas can reconcile.
            final_answer_text = accumulated
            yield "data: " + json.dumps(
                {
                    "type": "ai_message",
                    "id": ai_message_id or "ai_stream",
                    "content": accumulated,
                    "timestamp": None,
                }
            ) + "\n\n"

        # Write-through to the Q&A cache (notebook scope only). TRULY
        # fire-and-forget — we schedule the task and move on so the `complete`
        # SSE event flushes immediately. `save_cached_answer` internally
        # generates an embedding for the question (another LLM round-trip),
        # which we definitely don't want on the response's critical path.
        # Any exception inside the task is swallowed+logged by the helper.
        if (
            notebook_id
            and cache_fingerprint
            and final_answer_text.strip()
        ):
            try:
                asyncio.create_task(
                    save_cached_answer(
                        question=message,
                        answer=final_answer_text,
                        context_fingerprint=cache_fingerprint,
                        notebook_id=notebook_id,
                        model_id=model_override,
                    )
                )
            except Exception as e:
                # create_task() itself shouldn't raise, but guard just in case
                # so a scheduling hiccup never breaks the chat stream.
                logger.debug(f"Cache write scheduling failed (non-fatal): {e}")

        # Kick the notebook running summary — another fire-and-forget task
        # that only does real work once enough new messages have landed
        # (see REFRESH_BATCH_SIZE in notebook_summary.py). The summary is
        # injected into the chat system prompt next turn so the model
        # carries long-term memory across sessions.
        if notebook_id and final_answer_text.strip():
            try:
                from open_notebook.domain.notebook_summary import (
                    maybe_refresh_running_summary,
                )

                asyncio.create_task(
                    maybe_refresh_running_summary(
                        notebook_id, model_id=model_override
                    )
                )
            except Exception as e:
                logger.debug(
                    f"Running-summary scheduling failed (non-fatal): {e}"
                )

        # Completion signal
        yield "data: " + json.dumps({"type": "complete"}) + "\n\n"

    except Exception as e:
        from open_notebook.utils.error_classifier import classify_error

        try:
            _, user_message = classify_error(e)
        except Exception:
            user_message = str(e)
        logger.error(
            f"Error in chat streaming: {str(e)}\n{traceback.format_exc()}"
        )
        yield "data: " + json.dumps(
            {"type": "error", "message": user_message}
        ) + "\n\n"


@router.post("/chat/execute/stream")
async def execute_chat_stream(request: ExecuteChatRequest):
    """Execute a chat request and stream the AI response as SSE events.

    Drop-in streaming equivalent of /api/chat/execute. Emits:
      * user_message          (echo)
      * ai_message_delta      (token deltas)
      * ai_message            (final accumulated content)
      * context_indicators    (none for regular chat, source chat only)
      * complete              (end of stream)
      * error                 (on failure)
    """
    try:
        full_session_id = (
            request.session_id
            if request.session_id.startswith("chat_session:")
            else f"chat_session:{request.session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        model_override = (
            request.model_override
            if request.model_override is not None
            else getattr(session, "model_override", None)
        )

        if not request.message:
            raise HTTPException(status_code=400, detail="Message content is required")

        await session.save()

        # Resolve notebook scope for the Q&A cache. Prefer the value the
        # frontend already has (avoids a DB round-trip); fall back to the
        # session's refers_to edge so older clients still benefit.
        notebook_scope_id: Optional[str] = request.notebook_id
        if not notebook_scope_id:
            try:
                notebook_rows = await repo_query(
                    "SELECT out FROM refers_to WHERE in = $session_id",
                    {"session_id": ensure_record_id(full_session_id)},
                )
                if notebook_rows and notebook_rows[0].get("out"):
                    notebook_scope_id = str(notebook_rows[0]["out"])
            except Exception:
                # Cache is best-effort; missing scope just disables caching.
                notebook_scope_id = None

        return StreamingResponse(
            stream_chat_response(
                session_id=full_session_id,
                message=request.message,
                context=request.context,
                model_override=model_override,
                notebook_id=notebook_scope_id,
                bypass_cache=request.bypass_cache,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",  # disable proxy buffering
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting chat stream: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error starting chat stream: {str(e)}"
        )


@router.post("/chat/context", response_model=BuildContextResponse)
async def build_context(request: BuildContextRequest):
    """Build context for a notebook based on context configuration."""
    try:
        # Verify notebook exists
        notebook = await Notebook.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        context_data: dict[str, list[dict[str, str]]] = {"sources": [], "notes": []}
        total_content = ""

        # Process context configuration if provided
        if request.context_config:
            # Process sources
            for source_id, status in request.context_config.get("sources", {}).items():
                if "not in" in status:
                    continue

                try:
                    # Add table prefix if not present
                    full_source_id = (
                        source_id
                        if source_id.startswith("source:")
                        else f"source:{source_id}"
                    )

                    try:
                        source = await Source.get(full_source_id)
                    except Exception:
                        continue

                    if "insights" in status:
                        source_context = await source.get_context(context_size="short")
                        context_data["sources"].append(source_context)
                        total_content += str(source_context)
                    elif "full content" in status:
                        source_context = await source.get_context(context_size="long")
                        context_data["sources"].append(source_context)
                        total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source_id}: {str(e)}")
                    continue

            # Process notes
            for note_id, status in request.context_config.get("notes", {}).items():
                if "not in" in status:
                    continue

                try:
                    # Add table prefix if not present
                    full_note_id = (
                        note_id if note_id.startswith("note:") else f"note:{note_id}"
                    )
                    note = await Note.get(full_note_id)
                    if not note:
                        continue

                    if "full content" in status:
                        note_context = note.get_context(context_size="long")
                        context_data["notes"].append(note_context)
                        total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note {note_id}: {str(e)}")
                    continue
        else:
            # Default behavior - include all sources and notes with short context
            sources = await notebook.get_sources()
            for source in sources:
                try:
                    source_context = await source.get_context(context_size="short")
                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source.id}: {str(e)}")
                    continue

            notes = await notebook.get_notes()
            for note in notes:
                try:
                    note_context = note.get_context(context_size="short")
                    context_data["notes"].append(note_context)
                    total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note {note.id}: {str(e)}")
                    continue

        # Calculate character and token counts
        char_count = len(total_content)
        # Use token count utility if available
        try:
            from open_notebook.utils import token_count

            estimated_tokens = token_count(total_content) if total_content else 0
        except ImportError:
            # Fallback to simple estimation
            estimated_tokens = char_count // 4

        return BuildContextResponse(
            context=context_data, token_count=estimated_tokens, char_count=char_count
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building context: {str(e)}")
