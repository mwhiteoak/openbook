import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.chat_cache import (
    bump_cache_hit,
    find_cached_answer,
    save_cached_answer,
)
from open_notebook.domain.notebook import ChatSession, Source
from open_notebook.exceptions import (
    NotFoundError,
)
from open_notebook.graphs.source_chat import get_source_chat_graph
from open_notebook.utils.cache_fingerprint import compute_context_fingerprint
from open_notebook.utils.graph_utils import get_session_message_count

router = APIRouter()


# Request/Response models
class CreateSourceChatSessionRequest(BaseModel):
    source_id: str = Field(..., description="Source ID to create chat session for")
    title: Optional[str] = Field(None, description="Optional session title")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this session"
    )

class UpdateSourceChatSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )

class ChatMessage(BaseModel):
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (human|ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ContextIndicator(BaseModel):
    sources: List[str] = Field(
        default_factory=list, description="Source IDs used in context"
    )
    insights: List[str] = Field(
        default_factory=list, description="Insight IDs used in context"
    )
    notes: List[str] = Field(
        default_factory=list, description="Note IDs used in context"
    )

class SourceChatSessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    source_id: str = Field(..., description="Source ID")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )
    created: str = Field(..., description="Creation timestamp")
    updated: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(
        None, description="Number of messages in session"
    )

class SourceChatSessionWithMessagesResponse(SourceChatSessionResponse):
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Session messages"
    )
    context_indicators: Optional[ContextIndicator] = Field(
        None, description="Context indicators from last response"
    )

class SendMessageRequest(BaseModel):
    message: str = Field(..., description="User message content")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this message"
    )
    # Set by the UI's "Regenerate" button on a cached answer. We still write
    # the fresh answer to the cache, so the old entry is naturally superseded.
    bypass_cache: bool = Field(
        False, description="Skip Q&A cache lookup for this request"
    )

class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")


@router.post(
    "/sources/{source_id}/chat/sessions", response_model=SourceChatSessionResponse
)
async def create_source_chat_session(
    request: CreateSourceChatSessionRequest,
    source_id: str = Path(..., description="Source ID"),
):
    """Create a new chat session for a source."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Create new session with model_override support
        session = ChatSession(
            title=request.title or f"Source Chat {asyncio.get_event_loop().time():.0f}",
            model_override=request.model_override,
        )
        await session.save()

        # Relate session to source using "refers_to" relation
        await session.relate("refers_to", full_source_id)

        return SourceChatSessionResponse(
            id=session.id or "",
            title=session.title or "Untitled Session",
            source_id=source_id,
            model_override=session.model_override,
            created=str(session.created),
            updated=str(session.updated),
            message_count=0,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except Exception as e:
        logger.error(f"Error creating source chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating source chat session: {str(e)}"
        )


@router.get(
    "/sources/{source_id}/chat/sessions", response_model=List[SourceChatSessionResponse]
)
async def get_source_chat_sessions(source_id: str = Path(..., description="Source ID")):
    """Get all chat sessions for a source."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get sessions that refer to this source - first get relations, then sessions
        relations = await repo_query(
            "SELECT in FROM refers_to WHERE out = $source_id",
            {"source_id": ensure_record_id(full_source_id)},
        )

        sessions = []
        source_chat_graph = await get_source_chat_graph()
        for relation in relations:
            session_id_raw = relation.get("in")
            if session_id_raw:
                session_id = str(session_id_raw)

                session_result = await repo_query(
                    "SELECT * FROM $id", {"id": ensure_record_id(session_id)}
                )
                if session_result and len(session_result) > 0:
                    session_data = session_result[0]

                    # Get message count from LangGraph state
                    msg_count = await get_session_message_count(
                        source_chat_graph, session_id
                    )

                    sessions.append(
                        SourceChatSessionResponse(
                            id=session_data.get("id") or "",
                            title=session_data.get("title") or "Untitled Session",
                            source_id=source_id,
                            model_override=session_data.get("model_override"),
                            created=str(session_data.get("created")),
                            updated=str(session_data.get("updated")),
                            message_count=msg_count,
                        )
                    )

        # Sort sessions by created date (newest first)
        sessions.sort(key=lambda x: x.created, reverse=True)
        return sessions
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except Exception as e:
        logger.error(f"Error fetching source chat sessions: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching source chat sessions: {str(e)}"
        )


@router.get(
    "/sources/{source_id}/chat/sessions/{session_id}",
    response_model=SourceChatSessionWithMessagesResponse,
)
async def get_source_chat_session(
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
):
    """Get a specific source chat session with its messages."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get session
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session is related to this source
        relation_query = await repo_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )

        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        # Get session state from LangGraph to retrieve messages
        source_chat_graph = await get_source_chat_graph()
        thread_state = await source_chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": full_session_id}),
        )

        # Extract messages from state
        messages: list[ChatMessage] = []
        context_indicators = None

        if thread_state and thread_state.values:
            # Extract messages
            if "messages" in thread_state.values:
                for msg in thread_state.values["messages"]:
                    messages.append(
                        ChatMessage(
                            id=getattr(msg, "id", f"msg_{len(messages)}"),
                            type=msg.type if hasattr(msg, "type") else "unknown",
                            content=msg.content
                            if hasattr(msg, "content")
                            else str(msg),
                            timestamp=None,  # LangChain messages don't have timestamps by default
                        )
                    )

            # Extract context indicators from the last state
            if "context_indicators" in thread_state.values:
                context_data = thread_state.values["context_indicators"]
                context_indicators = ContextIndicator(
                    sources=context_data.get("sources", []),
                    insights=context_data.get("insights", []),
                    notes=context_data.get("notes", []),
                )

        return SourceChatSessionWithMessagesResponse(
            id=session.id or "",
            title=session.title or "Untitled Session",
            source_id=source_id,
            model_override=getattr(session, "model_override", None),
            created=str(session.created),
            updated=str(session.updated),
            message_count=len(messages),
            messages=messages,
            context_indicators=context_indicators,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Source or session not found")
    except Exception as e:
        logger.error(f"Error fetching source chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching source chat session: {str(e)}"
        )


@router.put(
    "/sources/{source_id}/chat/sessions/{session_id}",
    response_model=SourceChatSessionResponse,
)
async def update_source_chat_session(
    request: UpdateSourceChatSessionRequest,
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
):
    """Update source chat session title and/or model override."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get session
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session is related to this source
        relation_query = await repo_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )

        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        # Update session fields
        if request.title is not None:
            session.title = request.title
        if request.model_override is not None:
            session.model_override = request.model_override

        await session.save()

        # Get message count from LangGraph state
        source_chat_graph = await get_source_chat_graph()
        msg_count = await get_session_message_count(source_chat_graph, full_session_id)

        return SourceChatSessionResponse(
            id=session.id or "",
            title=session.title or "Untitled Session",
            source_id=source_id,
            model_override=getattr(session, "model_override", None),
            created=str(session.created),
            updated=str(session.updated),
            message_count=msg_count,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Source or session not found")
    except Exception as e:
        logger.error(f"Error updating source chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating source chat session: {str(e)}"
        )


@router.delete(
    "/sources/{source_id}/chat/sessions/{session_id}", response_model=SuccessResponse
)
async def delete_source_chat_session(
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
):
    """Delete a source chat session."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get session
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session is related to this source
        relation_query = await repo_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )

        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        await session.delete()

        return SuccessResponse(
            success=True, message="Source chat session deleted successfully"
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Source or session not found")
    except Exception as e:
        logger.error(f"Error deleting source chat session: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting source chat session: {str(e)}"
        )


async def _compute_source_chat_fingerprint(
    *,
    source_id: str,
    model_override: Optional[str],
) -> Optional[str]:
    """Fingerprint for source-chat cache entries.

    Source chats don't take a user-configurable context — the graph builds
    context internally from the source's insights + content. So the only
    moving parts are the source's ``updated`` timestamp (flips when content
    is re-ingested) and the selected model.
    """
    timestamps: List[Any] = []
    try:
        rows = await repo_query(
            "SELECT updated FROM $id",
            {"id": ensure_record_id(source_id)},
        )
        if rows and rows[0].get("updated"):
            timestamps.append(rows[0]["updated"])
    except Exception:
        return None

    return compute_context_fingerprint(
        notebook_id=None,
        source_id=source_id,
        context_config=None,
        model_id=model_override,
        source_updated_timestamps=timestamps,
    )


async def _stream_cached_source_chat_answer(
    *,
    session_id: str,
    message: str,
    cache_row: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Replay a cached answer over SSE + persist the turn to the checkpoint.

    Source chat has the same emission contract as notebook chat but with an
    extra `context_indicators` event — we intentionally skip emitting one
    here because the cached answer wasn't produced by the graph and we
    don't have a fresh context-indicators dict. The UI renders a "cached"
    badge instead, which is enough signal for the user.
    """
    answer_text = str(cache_row.get("answer") or "")
    cache_id = str(cache_row.get("id") or "")

    yield f"data: {json.dumps({'type': 'user_message', 'content': message, 'timestamp': None})}\n\n"

    ai_id = f"cached_{cache_id}" if cache_id else "cached"
    yield f"data: {json.dumps({'type': 'ai_message', 'id': ai_id, 'content': answer_text, 'timestamp': None, 'cached': True, 'cache_id': cache_id})}\n\n"

    # Persist user + AI turn into the source chat checkpoint so the
    # conversation history matches the live case. Failures here don't
    # affect the already-streamed response.
    try:
        source_chat_graph = await get_source_chat_graph()
        await source_chat_graph.aupdate_state(
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
        logger.warning(f"Failed to persist cached source-chat turn: {e}")

    if cache_id:
        await bump_cache_hit(cache_id)

    yield f"data: {json.dumps({'type': 'complete'})}\n\n"


async def stream_source_chat_response(
    session_id: str,
    source_id: str,
    message: str,
    model_override: Optional[str] = None,
    *,
    bypass_cache: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream the source chat response as Server-Sent Events.

    Uses LangGraph's `astream_events` to emit token-level deltas. Falls back
    to `ainvoke` if the provider doesn't support streaming. Consults the
    Q&A cache before invoking the graph to skip LLM calls for repeat
    questions within the same source + model scope.
    """
    # ---- Cache lookup (source scope) ------------------------------------
    cache_fingerprint = await _compute_source_chat_fingerprint(
        source_id=source_id,
        model_override=model_override,
    )

    if cache_fingerprint and not bypass_cache:
        try:
            cache_hit = await find_cached_answer(
                question=message,
                context_fingerprint=cache_fingerprint,
                source_id=source_id,
                model_id=model_override,
            )
        except Exception as e:
            logger.warning(f"Source-chat cache lookup raised: {e}")
            cache_hit = None

        if cache_hit:
            async for chunk in _stream_cached_source_chat_answer(
                session_id=session_id,
                message=message,
                cache_row=cache_hit,
            ):
                yield chunk
            return

    try:
        source_chat_graph = await get_source_chat_graph()
        # Get current checkpoint state via AsyncSqliteSaver
        current_state = await source_chat_graph.aget_state(
            config=RunnableConfig(configurable={"thread_id": session_id}),
        )

        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        state_values["source_id"] = source_id
        state_values["model_override"] = model_override

        user_message = HumanMessage(content=message)
        state_values["messages"].append(user_message)

        # Echo user message
        yield f"data: {json.dumps({'type': 'user_message', 'content': message, 'timestamp': None})}\n\n"

        accumulated = ""
        ai_message_id: Optional[str] = None
        saw_chunk = False
        final_state: dict = {}

        config = RunnableConfig(
            configurable={"thread_id": session_id, "model_id": model_override}
        )

        try:
            async for event in source_chat_graph.astream_events(
                input=state_values,  # type: ignore[arg-type]
                config=config,
                version="v2",
            ):
                ev_type = event.get("event")

                if ev_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is None:
                        continue
                    delta = getattr(chunk, "content", None)
                    if isinstance(delta, list):
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
                    yield f"data: {json.dumps({'type': 'ai_message_delta', 'id': ai_message_id, 'delta': delta})}\n\n"
                elif ev_type == "on_chain_end":
                    # Capture node-return state so we can emit context_indicators
                    data = event.get("data", {}) or {}
                    output = data.get("output")
                    if isinstance(output, dict) and "context_indicators" in output:
                        final_state = output
        except Exception as e:
            # Do NOT reset `saw_chunk` / `accumulated` here. If the stream
            # produced content before failing, LangGraph has already
            # checkpointed the AI message. Falling back to `ainvoke` would
            # run the LLM a second time and persist a duplicate AI message,
            # which the frontend's refetch then renders as two bubbles.
            logger.warning(
                f"Source chat stream interrupted after {len(accumulated)} "
                f"chars (saw_chunk={saw_chunk}): {e}"
            )

        final_answer_text: str = ""

        if not saw_chunk:
            # Fallback only when the stream produced nothing at all. Some
            # providers (esp. Ollama via certain proxies) don't emit
            # on_chat_model_stream events, so we still need a one-shot
            # ainvoke path. We emit only the NEWEST AI message — iterating
            # every AI message in the session history would echo past turns.
            result = await source_chat_graph.ainvoke(
                input=state_values,  # type: ignore[arg-type]
                config=config,
            )
            final_state = result or {}
            ai_msgs = [
                m
                for m in (result.get("messages") or [])
                if hasattr(m, "type") and m.type == "ai"
            ]
            if ai_msgs:
                latest = ai_msgs[-1]
                content = latest.content if hasattr(latest, "content") else str(latest)
                final_answer_text = str(content)
                yield f"data: {json.dumps({'type': 'ai_message', 'id': getattr(latest, 'id', 'ai_msg'), 'content': content, 'timestamp': None})}\n\n"
        else:
            final_answer_text = accumulated
            yield f"data: {json.dumps({'type': 'ai_message', 'id': ai_message_id or 'ai_stream', 'content': accumulated, 'timestamp': None})}\n\n"

        # Emit context indicators (sources/insights/notes referenced)
        context_indicators = final_state.get("context_indicators") if final_state else None
        if context_indicators is None:
            # Fall back to checkpoint state
            try:
                latest = await source_chat_graph.aget_state(
                    config=RunnableConfig(configurable={"thread_id": session_id}),
                )
                if latest and latest.values:
                    context_indicators = latest.values.get("context_indicators")
            except Exception:
                context_indicators = None

        if context_indicators:
            yield f"data: {json.dumps({'type': 'context_indicators', 'data': context_indicators})}\n\n"

        # Write-through to Q&A cache. TRULY fire-and-forget: schedule it and
        # move on so the `complete` SSE event flushes immediately. The helper
        # generates an embedding for the question internally (another LLM
        # round-trip), which we don't want on the user-visible response path.
        if cache_fingerprint and final_answer_text.strip():
            try:
                asyncio.create_task(
                    save_cached_answer(
                        question=message,
                        answer=final_answer_text,
                        context_fingerprint=cache_fingerprint,
                        source_id=source_id,
                        model_id=model_override,
                    )
                )
            except Exception as e:
                logger.debug(f"Source-chat cache write scheduling failed (non-fatal): {e}")

        # Completion
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    except Exception as e:
        from open_notebook.utils.error_classifier import classify_error

        try:
            _, user_message = classify_error(e)
        except Exception:
            user_message = str(e)
        logger.error(f"Error in source chat streaming: {str(e)}")
        yield f"data: {json.dumps({'type': 'error', 'message': user_message})}\n\n"


@router.post("/sources/{source_id}/chat/sessions/{session_id}/messages")
async def send_message_to_source_chat(
    request: SendMessageRequest,
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
):
    """Send a message to source chat session with SSE streaming response."""
    try:
        # Verify source exists
        full_source_id = (
            source_id if source_id.startswith("source:") else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Verify session exists and is related to source
        full_session_id = (
            session_id
            if session_id.startswith("chat_session:")
            else f"chat_session:{session_id}"
        )
        session = await ChatSession.get(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session is related to this source
        relation_query = await repo_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )

        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        if not request.message:
            raise HTTPException(status_code=400, detail="Message content is required")

        # Determine model override (request override takes precedence over session override)
        model_override = request.model_override or getattr(
            session, "model_override", None
        )

        # Update session timestamp
        await session.save()

        # Return streaming response
        return StreamingResponse(
            stream_source_chat_response(
                session_id=full_session_id,
                source_id=full_source_id,
                message=request.message,
                model_override=model_override,
                bypass_cache=request.bypass_cache,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to source chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")
