'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { getApiErrorMessage } from '@/lib/utils/error-handler'
import { useTranslation } from '@/lib/hooks/use-translation'
import { sourceChatApi } from '@/lib/api/source-chat'
import {
  SourceChatSession,
  SourceChatMessage,
  SourceChatContextIndicator,
  CreateSourceChatSessionRequest,
  UpdateSourceChatSessionRequest,
  RetrievalSource
} from '@/lib/types/api'

export function useSourceChat(sourceId: string) {
  const { t } = useTranslation()
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<SourceChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [contextIndicators, setContextIndicators] = useState<SourceChatContextIndicator | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  // Fetch sessions
  const { data: sessions = [], isLoading: loadingSessions, refetch: refetchSessions } = useQuery<SourceChatSession[]>({
    queryKey: ['sourceChatSessions', sourceId],
    queryFn: () => sourceChatApi.listSessions(sourceId),
    enabled: !!sourceId
  })

  // Fetch current session with messages
  const { data: currentSession, refetch: refetchCurrentSession } = useQuery({
    queryKey: ['sourceChatSession', sourceId, currentSessionId],
    queryFn: () => sourceChatApi.getSession(sourceId, currentSessionId!),
    enabled: !!sourceId && !!currentSessionId
  })

  // Update messages when session changes
  useEffect(() => {
    if (currentSession?.messages) {
      setMessages(currentSession.messages)
    }
  }, [currentSession])

  // Auto-select most recent session when sessions are loaded
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      // Find most recent session (sessions are sorted by created date desc from API)
      const mostRecentSession = sessions[0]
      setCurrentSessionId(mostRecentSession.id)
    }
  }, [sessions, currentSessionId])

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: Omit<CreateSourceChatSessionRequest, 'source_id'>) => 
      sourceChatApi.createSession(sourceId, data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ['sourceChatSessions', sourceId] })
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
    mutationFn: ({ sessionId, data }: { sessionId: string, data: UpdateSourceChatSessionRequest }) =>
      sourceChatApi.updateSession(sourceId, sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sourceChatSessions', sourceId] })
      queryClient.invalidateQueries({ queryKey: ['sourceChatSession', sourceId, currentSessionId] })
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
      sourceChatApi.deleteSession(sourceId, sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ['sourceChatSessions', sourceId] })
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

  // Send message with streaming.
  //
  // Same optimistic flow as notebook chat:
  //   1. Push user's message + AI placeholder (isStreaming: true) upfront.
  //   2. Accumulate `ai_message_delta` chunks onto the placeholder.
  //   3. On `ai_message` (final), replace placeholder content; stamp
  //      `cached`/`cacheId`/`prompt` when the event came from the Q&A cache
  //      so ChatPanel can render the "⚡ Cached · Regenerate" badge.
  //   4. Clear isStreaming on `complete` (or on final ai_message).
  //
  // The `options.bypassCache` flag is set by the Regenerate button; backend
  // skips cache lookup but still writes through so the fresh answer wins.
  const sendMessage = useCallback(async (
    message: string,
    modelOverride?: string,
    options?: { bypassCache?: boolean }
  ) => {
    let sessionId = currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30 ? `${message.substring(0, 30)}...` : message
        const newSession = await sourceChatApi.createSession(sourceId, { title: defaultTitle })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        queryClient.invalidateQueries({ queryKey: ['sourceChatSessions', sourceId] })
      } catch (err: unknown) {
        const error = err as { response?: { data?: { detail?: string } }, message?: string };
        console.error('Failed to create chat session:', error)
        toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToCreateSession'))
        return
      }
    }

    // Optimistic user message + AI placeholder with streaming cursor
    const now = Date.now()
    const userTempId = `temp-user-${now}`
    const aiTempId = `temp-ai-${now}`
    const userMessage: SourceChatMessage = {
      id: userTempId,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    const aiPlaceholder: SourceChatMessage = {
      id: aiTempId,
      type: 'ai',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true
    }
    setMessages(prev => [...prev, userMessage, aiPlaceholder])
    setIsStreaming(true)

    let accumulated = ''
    let sawAnyChunk = false

    try {
      const response = await sourceChatApi.sendMessage(sourceId, sessionId, {
        message,
        model_override: modelOverride,
        bypass_cache: options?.bypassCache ?? false
      })

      if (!response) {
        throw new Error('No response body')
      }

      const reader = response.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        // Keep the trailing partial chunk in buffer.
        buffer = lines.pop() ?? ''

        for (const rawLine of lines) {
          const line = rawLine.trim()
          if (!line.startsWith('data: ')) continue

          let data: {
            type: string
            delta?: string
            content?: string
            message?: string
            id?: string
            data?: unknown
            cached?: boolean
            cache_id?: string
            // Retrieval preview payload — present on `retrieval_sources`
            // events emitted after the retrieve node runs.
            sources?: RetrievalSource[]
          }
          try {
            data = JSON.parse(line.slice(6))
          } catch (parseErr) {
            console.error('Error parsing SSE data:', parseErr)
            continue
          }

          if (data.type === 'retrieval_sources' && Array.isArray(data.sources)) {
            // Stamp the AI placeholder with the retrieval preview so the
            // UI can render source chips above the bubble before the first
            // answer token arrives. Shared semantics with useNotebookChat.
            const retrieval = data.sources
            setMessages(prev =>
              prev.map(m => m.id === aiTempId ? { ...m, retrievalSources: retrieval } : m)
            )
          } else if (data.type === 'ai_message_delta' && typeof data.delta === 'string') {
            sawAnyChunk = true
            accumulated += data.delta
            setMessages(prev =>
              prev.map(m => m.id === aiTempId ? { ...m, content: accumulated } : m)
            )
          } else if (data.type === 'ai_message' && typeof data.content === 'string') {
            // Final content (or non-streaming fallback); reconcile.
            sawAnyChunk = true
            accumulated = data.content
            const isCached = data.cached === true
            const cacheIdVal = data.cache_id ?? undefined
            setMessages(prev =>
              prev.map(m =>
                m.id === aiTempId
                  ? {
                      ...m,
                      content: accumulated,
                      isStreaming: false,
                      ...(isCached ? { cached: true, cacheId: cacheIdVal } : {})
                    }
                  : m
              )
            )
          } else if (data.type === 'context_indicators') {
            setContextIndicators(data.data as SourceChatContextIndicator)
          } else if (data.type === 'complete') {
            // Clear streaming flag in case only deltas arrived.
            setMessages(prev =>
              prev.map(m => m.id === aiTempId ? { ...m, isStreaming: false } : m)
            )
          } else if (data.type === 'error') {
            throw new Error(data.message || 'Stream error')
          }
        }
      }

      if (!sawAnyChunk) {
        throw new Error('Empty response from chat stream')
      }
    } catch (err: unknown) {
      const error = err as { response?: { data?: { detail?: string } }, message?: string };
      console.error('Error sending message:', error)
      toast.error(getApiErrorMessage(error.response?.data?.detail || error.message, (key) => t(key), 'apiErrors.failedToSendMessage'))
      // Remove optimistic pair on error
      setMessages(prev => prev.filter(msg => msg.id !== userTempId && msg.id !== aiTempId))
    } finally {
      setIsStreaming(false)
      // Refetch session to get persisted messages
      refetchCurrentSession()
    }
  }, [sourceId, currentSessionId, refetchCurrentSession, queryClient, t])

  // Regenerate a cached answer. Finds the question that produced the cached
  // bubble (preferring the captured prompt, falling back to the preceding
  // human turn) and re-sends it with bypass_cache=true.
  const regenerateMessage = useCallback(
    async (messageId: string) => {
      const target = messages.find(m => m.id === messageId)
      if (!target || target.type !== 'ai') return
      const idx = messages.findIndex(m => m.id === messageId)
      let prompt: string | undefined
      for (let i = idx - 1; i >= 0; i--) {
        if (messages[i].type === 'human') {
          prompt = messages[i].content
          break
        }
      }
      if (!prompt) return
      await sendMessage(prompt, undefined, { bypassCache: true })
    },
    [messages, sendMessage]
  )

  // Cancel streaming
  const cancelStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsStreaming(false)
    }
  }, [])

  // Switch session
  const switchSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
    setContextIndicators(null)
  }, [])

  // Create session
  const createSession = useCallback((data: Omit<CreateSourceChatSessionRequest, 'source_id'>) => {
    return createSessionMutation.mutate(data)
  }, [createSessionMutation])

  // Update session
  const updateSession = useCallback((sessionId: string, data: UpdateSourceChatSessionRequest) => {
    return updateSessionMutation.mutate({ sessionId, data })
  }, [updateSessionMutation])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    return deleteSessionMutation.mutate(sessionId)
  }, [deleteSessionMutation])

  return {
    // State
    sessions,
    currentSession: sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages,
    isStreaming,
    contextIndicators,
    loadingSessions,
    
    // Actions
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    regenerateMessage,
    cancelStreaming,
    refetchSessions
  }
}
