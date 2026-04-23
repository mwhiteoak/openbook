export interface NotebookResponse {
  id: string
  name: string
  description: string
  archived: boolean
  created: string
  updated: string
  source_count: number
  note_count: number
}

export interface NoteResponse {
  id: string
  title: string | null
  content: string | null
  note_type: string | null
  created: string
  updated: string
}

export interface SourceListResponse {
  id: string
  title: string | null
  topics?: string[]                  // Make optional to match Python API
  asset: {
    file_path?: string
    url?: string
  } | null
  embedded: boolean
  embedded_chunks: number            // ADD: From Python API
  insights_count: number
  created: string
  updated: string
  file_available?: boolean
  // ADD: Async processing fields from Python API
  command_id?: string
  status?: string
  processing_info?: Record<string, unknown>
}

export interface SourceDetailResponse extends SourceListResponse {
  full_text: string
  notebooks?: string[]  // List of notebook IDs this source is linked to
}

export type SourceResponse = SourceDetailResponse

export interface SourceStatusResponse {
  status?: string
  message: string
  processing_info?: Record<string, unknown>
  command_id?: string
}

export interface SettingsResponse {
  default_content_processing_engine_doc?: string
  default_content_processing_engine_url?: string
  default_embedding_option?: string
  auto_delete_files?: string
  youtube_preferred_languages?: string[]
}

export interface CreateNotebookRequest {
  name: string
  description?: string
}

export interface UpdateNotebookRequest {
  name?: string
  description?: string
  archived?: boolean
}

export interface NotebookDeletePreview {
  notebook_id: string
  notebook_name: string
  note_count: number
  exclusive_source_count: number
  shared_source_count: number
}

export interface NotebookDeleteResponse {
  message: string
  deleted_notes: number
  deleted_sources: number
  unlinked_sources: number
}

export interface CreateNoteRequest {
  title?: string
  content: string
  note_type?: string
  notebook_id?: string
}

export interface CreateSourceRequest {
  // Backward compatibility: support old single notebook_id
  notebook_id?: string
  // New multi-notebook support
  notebooks?: string[]
  // Required fields
  type: 'link' | 'upload' | 'text'
  url?: string
  file_path?: string
  content?: string
  title?: string
  transformations?: string[]
  embed?: boolean
  delete_source?: boolean
  // New async processing support
  async_processing?: boolean
}

export interface UpdateNoteRequest {
  title?: string
  content?: string
  note_type?: string
}

export interface UpdateSourceRequest {
  title?: string
  type?: 'link' | 'upload' | 'text'
  url?: string
  content?: string
}

export interface APIError {
  detail: string
}

// Source Chat Types
// Base session interface with common fields
export interface BaseChatSession {
  id: string
  title: string
  created: string
  updated: string
  message_count?: number
  model_override?: string | null
}

export interface SourceChatSession extends BaseChatSession {
  source_id: string
  model_override?: string
}

export interface SourceChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
  // True while SSE deltas are still arriving (shared semantics with
  // NotebookChatMessage). ChatPanel uses this to drive a streaming cursor
  // and to suppress the standalone "thinking…" bubble.
  isStreaming?: boolean
  // Set when the answer came from the Q&A cache rather than a fresh LLM
  // call. The UI renders a "⚡ Cached · Regenerate" badge on these bubbles.
  cached?: boolean
  // Cache row id — used by the Regenerate button (which sends the same
  // question with bypass_cache=true, causing a fresh answer to overwrite
  // the old one on the next lookup).
  cacheId?: string
  // Perplexity-style retrieval preview. Populated by the `retrieval_sources`
  // SSE event emitted after the retrieve node runs but before the answer
  // streams in. Present on notebook chat and (once wired) source chat.
  retrievalSources?: RetrievalSource[]
}

export interface SourceChatContextIndicator {
  sources: string[]
  insights: string[]
  notes: string[]
}

export interface SourceChatSessionWithMessages extends SourceChatSession {
  messages: SourceChatMessage[]
  context_indicators?: SourceChatContextIndicator
}

export interface CreateSourceChatSessionRequest {
  source_id: string
  title?: string
  model_override?: string
}

export interface UpdateSourceChatSessionRequest {
  title?: string
  model_override?: string
}

export interface SendMessageRequest {
  message: string
  model_override?: string
  // True when the user clicked "Regenerate" on a cached answer. Backend
  // skips cache lookup but still writes through so the new answer supersedes.
  bypass_cache?: boolean
}

export interface SourceChatStreamEvent {
  type:
    | 'user_message'
    | 'ai_message'
    | 'ai_message_delta'
    | 'context_indicators'
    | 'complete'
    | 'error'
  id?: string
  delta?: string
  content?: string
  data?: unknown
  message?: string
  timestamp?: string
  // Present on ai_message events served from the Q&A cache.
  cached?: boolean
  cache_id?: string
}

// Notebook Chat Types
export interface NotebookChatSession extends BaseChatSession {
  notebook_id: string
}

export interface NotebookChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
  // Set to true while SSE deltas are still arriving for this message.
  // The UI renders a typing cursor on streaming bubbles and suppresses the
  // standalone "AI is thinking…" spinner when a streaming AI message exists.
  isStreaming?: boolean
  // Set when this answer was served from the Q&A cache. Drives the
  // "⚡ Cached · Regenerate" badge in ChatPanel.
  cached?: boolean
  // Server-side cache row id; needed so the Regenerate button can link the
  // new LLM answer back to the stored row (the backend overwrites by
  // fingerprint+question, so we just re-send with bypass_cache).
  cacheId?: string
  // Original question that produced this answer. We need it for the
  // Regenerate click because the AI bubble itself doesn't carry the prompt.
  prompt?: string
  // Perplexity-style retrieval preview: light-weight descriptor of the
  // source excerpts that fed the LLM for this turn. Populated by the
  // `retrieval_sources` SSE event, which fires AFTER the retrieve node
  // runs but BEFORE any answer tokens stream in. The UI renders these as
  // chips above the answer so users see what was searched before the
  // answer is even written.
  retrievalSources?: RetrievalSource[]
}

// Descriptor of a source chunk surfaced by hybrid retrieval. Emitted in the
// `retrieval_sources` SSE event — see open_notebook/graphs/chat.py and
// api/routers/chat.py for producer details.
export interface RetrievalSource {
  source_id: string
  title: string
  similarity: number
  matched_vector: boolean
  matched_text: boolean
}

export interface NotebookChatSessionWithMessages extends NotebookChatSession {
  messages: NotebookChatMessage[]
}

export interface CreateNotebookChatSessionRequest {
  notebook_id: string
  title?: string
  model_override?: string
}

export interface UpdateNotebookChatSessionRequest {
  title?: string
  model_override?: string | null
}

export interface SendNotebookChatMessageRequest {
  session_id: string
  message: string
  context: {
    sources: Array<Record<string, unknown>>
    notes: Array<Record<string, unknown>>
  }
  model_override?: string
  // True when the user clicked Regenerate on a cached answer — the backend
  // skips the cache lookup but still writes the fresh answer, which
  // supersedes the cached row on the next hit.
  bypass_cache?: boolean
  // Scope for the Q&A cache lookup. Passing this saves the API a refers_to
  // round-trip; older clients without it fall back to the session edge.
  notebook_id?: string
}

export interface BuildContextRequest {
  notebook_id: string
  context_config: {
    sources: Record<string, string>
    notes: Record<string, string>
  }
}

export interface BuildContextResponse {
  context: {
    sources: Array<Record<string, unknown>>
    notes: Array<Record<string, unknown>>
  }
  token_count: number
  char_count: number
}
