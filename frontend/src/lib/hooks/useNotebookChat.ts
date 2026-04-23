'use client'

import { useState, useCallback, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { getApiErrorMessage } from '@/lib/utils/error-handler'
import { useTranslation } from '@/lib/hooks/use-translation'
import { chatApi } from '@/lib/api/chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import {
  NotebookChatMessage,
  CreateNotebookChatSessionRequest,
  UpdateNotebookChatSessionRequest,
  SourceListResponse,
  NoteResponse,
  RetrievalSource
} from '@/lib/types/api'
import { ContextSelections } from '@/app/(dashboard)/notebooks/[id]/page'

interface UseNotebookChatParams {
  notebookId: string
  sources: SourceListResponse[]
  notes: NoteResponse[]
  contextSelections: ContextSelections
}

export function useNotebookChat({ notebookId, sources, notes, contextSelections }: UseNotebookChatParams) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<NotebookChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [tokenCount, setTokenCount] = useState<number>(0)
  const [charCount, setCharCount] = useState<number>(0)
  // Pending model override for when user changes model before a session exists
  const [pendingModelOverride, setPendingModelOverride] = useState<string | null>(null)

  // Fetch sessions for this notebook
  const {
    data: sessions = [],
    isLoading: loadingSessions,
    refetch: refetchSessions
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSessions(notebookId),
    queryFn: () => chatApi.listSessions(notebookId),
    enabled: !!notebookId
  })

  // Fetch current session with messages
  const {
    data: currentSession,
    refetch: refetchCurrentSession
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!),
    queryFn: () => chatApi.getSession(currentSessionId!),
    enabled: !!notebookId && !!currentSessionId
  })

  // Update messages when current session changes
  useEffect(() => {
    if (currentSession?.messages) {
      setMessages(currentSession.messages)
    }
  }, [currentSession])

  // Auto-select most recent session when sessions are loaded
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      // Sessions are sorted by created date desc from API
      const mostRecentSession = sessions[0]
      setCurrentSessionId(mostRecentSession.id)
    }
  }, [sessions, currentSessionId])

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: CreateNotebookChatSessionRequest) =>
      chatApi.createSession(data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      setCurrentSessionId(newSession.id)
      toast.success(t('chat.sessionCreated'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
    }
  })

  // Update session mutation
  const updateSessionMutation = useMutation({
    mutationFn: ({ sessionId, data }: {
      sessionId: string
      data: UpdateNotebookChatSessionRequest
    }) => chatApi.updateSession(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!)
      })
      toast.success(t('chat.sessionUpdated'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToUpdateSession'))
    }
  })

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) =>
      chatApi.deleteSession(sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      if (currentSessionId === deletedId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      toast.success(t('chat.sessionDeleted'))
    },
    onError: (err: unknown) => {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToDeleteSession'))
    }
  })

  // Build context from sources and notes based on user selections
  const buildContext = useCallback(async () => {
    // Build context_config mapping IDs to selection modes
    const context_config: { sources: Record<string, string>, notes: Record<string, string> } = {
      sources: {},
      notes: {}
    }

    // Map source selections
    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') {
        context_config.sources[source.id] = 'insights'
      } else if (mode === 'full') {
        context_config.sources[source.id] = 'full content'
      } else {
        context_config.sources[source.id] = 'not in'
      }
    })

    // Map note selections
    notes.forEach(note => {
      const mode = contextSelections.notes[note.id]
      if (mode === 'full') {
        context_config.notes[note.id] = 'full content'
      } else {
        context_config.notes[note.id] = 'not in'
      }
    })

    // Call API to build context with actual content
    const response = await chatApi.buildContext({
      notebook_id: notebookId,
      context_config
    })

    // Store token and char counts
    setTokenCount(response.token_count)
    setCharCount(response.char_count)

    return response.context
  }, [notebookId, sources, notes, contextSelections])

  // Send message with SSE streaming.
  //
  // Optimistic UX flow:
  //   1. Push the user's message immediately (temp- id).
  //   2. Push an AI placeholder with isStreaming: true so the typing cursor
  //      appears on an actual bubble instead of a detached spinner.
  //   3. As `ai_message_delta` events arrive, append deltas to the placeholder.
  //   4. On `ai_message` (final), replace placeholder content and clear
  //      isStreaming so the cursor goes away. If the event carries `cached:
  //      true` we stamp the message with `cached`/`cacheId` so ChatPanel
  //      renders the "⚡ Cached · Regenerate" badge.
  //   5. On `complete`, refetch the session so ids/timestamps match the server.
  //   6. On error, drop the optimistic pair so the user can retry cleanly.
  const sendMessage = useCallback(async (message: string, modelOverride?: string, options?: { bypassCache?: boolean }) => {
    let sessionId = currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30
          ? `${message.substring(0, 30)}...`
          : message
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: defaultTitle,
          // Include pending model override when creating session
          model_override: pendingModelOverride ?? undefined
        })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        // Clear pending model override now that it's applied to the session
        setPendingModelOverride(null)
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
        })
      } catch (err: unknown) {
        const error = err as { response?: { data?: { detail?: string } }, message?: string };
        toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
        return
      }
    }

    // Add optimistic user message + AI placeholder.
    const now = Date.now()
    const userTempId = `temp-user-${now}`
    const aiTempId = `temp-ai-${now}`
    const userMessage: NotebookChatMessage = {
      id: userTempId,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    const aiPlaceholder: NotebookChatMessage = {
      id: aiTempId,
      type: 'ai',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true
    }
    setMessages(prev => [...prev, userMessage, aiPlaceholder])
    setIsSending(true)

    let accumulated = ''
    let sawAnyChunk = false

    try {
      // Build context and open stream
      const context = await buildContext()
      const body = await chatApi.streamMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (currentSession?.model_override ?? undefined),
        bypass_cache: options?.bypassCache ?? false,
        notebook_id: notebookId,
      })

      if (!body) {
        throw new Error('No response body')
      }

      const reader = body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // SSE events are terminated by a blank line; split by \n and process
        // complete `data: ...` lines, keeping any partial trailing chunk.
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const rawLine of lines) {
          const line = rawLine.trim()
          if (!line.startsWith('data: ')) continue

          let evt: {
            type: string
            delta?: string
            content?: string
            message?: string
            id?: string
            timestamp?: string
            // Present on ai_message events served from the Q&A cache. We
            // propagate these onto the message so the UI can render the
            // "⚡ Cached" badge + Regenerate button.
            cached?: boolean
            cache_id?: string
            // Present on retrieval_sources events emitted right after the
            // retrieve node finishes (before any answer tokens stream in).
            sources?: RetrievalSource[]
          }
          try {
            evt = JSON.parse(line.slice(6))
          } catch (parseErr) {
            console.error('Error parsing SSE data:', parseErr)
            continue
          }

          if (evt.type === 'retrieval_sources' && Array.isArray(evt.sources)) {
            // Stamp the still-streaming AI placeholder with the sources that
            // were surfaced by hybrid retrieval. This lets ChatPanel render
            // "Searching N sources…" chips ABOVE the answer as soon as
            // retrieval finishes — users see progress before any tokens flow.
            const sources = evt.sources
            setMessages(prev =>
              prev.map(m =>
                m.id === aiTempId ? { ...m, retrievalSources: sources } : m
              )
            )
          } else if (evt.type === 'ai_message_delta' && typeof evt.delta === 'string') {
            sawAnyChunk = true
            accumulated += evt.delta
            // Functional update so React state updates interleave cleanly
            // with rapid delta bursts.
            setMessages(prev =>
              prev.map(m =>
                m.id === aiTempId ? { ...m, content: accumulated } : m
              )
            )
          } else if (evt.type === 'ai_message' && typeof evt.content === 'string') {
            // Final content; reconcile (covers the non-streaming fallback path
            // on the server where only ai_message is emitted).
            sawAnyChunk = true
            accumulated = evt.content
            const isCached = evt.cached === true
            const cacheIdVal = evt.cache_id ?? undefined
            setMessages(prev =>
              prev.map(m =>
                m.id === aiTempId
                  ? {
                      ...m,
                      content: accumulated,
                      isStreaming: false,
                      // Only set cached fields when present; leaves live
                      // answers untouched.
                      ...(isCached ? { cached: true, cacheId: cacheIdVal, prompt: message } : {})
                    }
                  : m
              )
            )
          } else if (evt.type === 'error') {
            throw new Error(evt.message || 'Stream error')
          } else if (evt.type === 'complete') {
            // Clear streaming flag in case only deltas arrived and no final
            // ai_message event was emitted.
            setMessages(prev =>
              prev.map(m =>
                m.id === aiTempId ? { ...m, isStreaming: false } : m
              )
            )
          }
          // user_message echo is ignored; we already rendered the user bubble.
        }
      }

      if (!sawAnyChunk) {
        // Server finished without producing any AI content — treat as failure
        // so the user isn't left with a blank bubble.
        throw new Error('Empty response from chat stream')
      }

      // Refetch to reconcile temp ids with real db ids/timestamps.
      await refetchCurrentSession()
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      console.error('Error sending message:', error)
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToSendMessage'))
      // Remove both optimistic messages on error so the user can retry.
      setMessages(prev => prev.filter(msg => msg.id !== userTempId && msg.id !== aiTempId))
    } finally {
      setIsSending(false)
    }
  }, [
    notebookId,
    currentSessionId,
    currentSession,
    pendingModelOverride,
    buildContext,
    refetchCurrentSession,
    queryClient,
    t
  ])

  // Regenerate a cached answer. The UI calls this when the user clicks the
  // "Regenerate" button on a cached bubble — we re-send the original question
  // with bypass_cache so the backend runs the LLM fresh (and then writes the
  // new answer to the cache, superseding the old one on the next hit).
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      const target = messages.find((m) => m.id === messageId)
      if (!target || target.type !== 'ai') return
      // Prefer the prompt we captured at cache-serve time. If we don't have
      // it (shouldn't happen, but be defensive), fall back to the preceding
      // human turn in the transcript.
      let prompt = target.prompt
      if (!prompt) {
        const idx = messages.findIndex((m) => m.id === messageId)
        for (let i = idx - 1; i >= 0; i--) {
          if (messages[i].type === 'human') {
            prompt = messages[i].content
            break
          }
        }
      }
      if (!prompt) return
      await sendMessage(prompt, undefined, { bypassCache: true })
    },
    [messages, sendMessage]
  )

  // Switch session
  const switchSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
  }, [])

  // Create session
  const createSession = useCallback((title?: string) => {
    return createSessionMutation.mutate({
      notebook_id: notebookId,
      title
    })
  }, [createSessionMutation, notebookId])

  // Update session
  const updateSession = useCallback((sessionId: string, data: UpdateNotebookChatSessionRequest) => {
    return updateSessionMutation.mutate({
      sessionId,
      data
    })
  }, [updateSessionMutation])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    return deleteSessionMutation.mutate(sessionId)
  }, [deleteSessionMutation])

  // Set model override - handles both existing sessions and pending state
  const setModelOverride = useCallback((model: string | null) => {
    if (currentSessionId) {
      // Session exists - update it directly
      updateSessionMutation.mutate({
        sessionId: currentSessionId,
        data: { model_override: model }
      })
    } else {
      // No session yet - store as pending
      setPendingModelOverride(model)
    }
  }, [currentSessionId, updateSessionMutation])

  // Update token/char counts when context selections change
  useEffect(() => {
    const updateContextCounts = async () => {
      try {
        await buildContext()
      } catch (error) {
        console.error('Error updating context counts:', error)
      }
    }
    updateContextCounts()
  }, [buildContext])

  return {
    // State
    sessions,
    currentSession: currentSession || sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages,
    isSending,
    loadingSessions,
    tokenCount,
    charCount,
    pendingModelOverride,

    // Actions
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    regenerateMessage,
    setModelOverride,
    refetchSessions
  }
}
