'use client'

import { useRouter } from 'next/navigation'
import { useQueryClient } from '@tanstack/react-query'
import { NotebookResponse, NoteResponse, SourceListResponse } from '@/lib/types/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { MoreHorizontal, Archive, ArchiveRestore, Trash2, FileText, StickyNote } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useUpdateNotebook } from '@/lib/hooks/use-notebooks'
import { NotebookDeleteDialog } from './NotebookDeleteDialog'
import { useRef, useState } from 'react'
import { useTranslation } from '@/lib/hooks/use-translation'
import { getDateLocale } from '@/lib/utils/date-locale'
import { notebooksApi } from '@/lib/api/notebooks'
import { notesApi } from '@/lib/api/notes'
import { sourcesApi } from '@/lib/api/sources'
import { QUERY_KEYS } from '@/lib/api/query-client'

// Match the page size used by useNotebookSources; keeping these in sync
// ensures the prefetched infinite-query page is reused without a refetch.
const NOTEBOOK_SOURCES_PAGE_SIZE = 30
interface NotebookCardProps {
  notebook: NotebookResponse
}

export function NotebookCard({ notebook }: NotebookCardProps) {
  const { t, language } = useTranslation()
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const router = useRouter()
  const queryClient = useQueryClient()
  const updateNotebook = useUpdateNotebook()
  // Fire prefetch at most once per card lifetime to avoid hammering the API
  // on incidental mouse movement (hover in/out repeatedly).
  const hasPrefetchedRef = useRef(false)

  const handleArchiveToggle = (e: React.MouseEvent) => {
    e.stopPropagation()
    updateNotebook.mutate({
      id: notebook.id,
      data: { archived: !notebook.archived }
    })
  }

  const notebookHref = `/notebooks/${encodeURIComponent(notebook.id)}`

  const handleCardClick = () => {
    router.push(notebookHref)
  }

  // Warm both the Next.js route cache and TanStack Query cache on hover so
  // that clicking the card renders instantly. We run the data prefetches in
  // parallel and swallow failures — this is a latency hint, not a correctness
  // requirement, and the target page will re-fetch on mount if anything is
  // still missing.
  const handlePrefetch = () => {
    if (hasPrefetchedRef.current) return
    hasPrefetchedRef.current = true

    // Next.js route bundle + RSC payload
    try {
      router.prefetch(notebookHref)
    } catch {
      // router.prefetch throws on some test environments; ignore.
    }

    // Keep these in sync with the queries used by the notebook page:
    //   - useNotebook(id)           -> QUERY_KEYS.notebook
    //   - useNotebookSources(id)    -> QUERY_KEYS.sourcesInfinite (infinite)
    //   - useNotes(notebookId)      -> QUERY_KEYS.notes
    queryClient.prefetchQuery({
      queryKey: QUERY_KEYS.notebook(notebook.id),
      queryFn: () => notebooksApi.get(notebook.id),
      staleTime: 30 * 1000,
    }).catch(() => { /* silent */ })

    queryClient.prefetchInfiniteQuery({
      queryKey: QUERY_KEYS.sourcesInfinite(notebook.id),
      queryFn: async ({ pageParam = 0 }: { pageParam?: number }) => {
        const data = await sourcesApi.list({
          notebook_id: notebook.id,
          limit: NOTEBOOK_SOURCES_PAGE_SIZE,
          offset: pageParam,
          sort_by: 'updated',
          sort_order: 'desc',
        })
        return {
          sources: data as SourceListResponse[],
          nextOffset:
            data.length === NOTEBOOK_SOURCES_PAGE_SIZE
              ? pageParam + data.length
              : undefined,
        }
      },
      initialPageParam: 0,
      staleTime: 30 * 1000,
    }).catch(() => { /* silent */ })

    queryClient.prefetchQuery({
      queryKey: QUERY_KEYS.notes(notebook.id),
      queryFn: () =>
        notesApi.list({ notebook_id: notebook.id }) as Promise<NoteResponse[]>,
      staleTime: 30 * 1000,
    }).catch(() => { /* silent */ })
  }

  return (
    <>
      <Card
        className="group card-hover"
        onClick={handleCardClick}
        onMouseEnter={handlePrefetch}
        onFocus={handlePrefetch}
        style={{ cursor: 'pointer' }}
      >
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <CardTitle className="text-base truncate group-hover:text-primary transition-colors">
                  {notebook.name}
                </CardTitle>
                {notebook.archived && (
                  <Badge variant="secondary" className="mt-1">
                    {t('notebooks.archived')}
                  </Badge>
                )}
              </div>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <MoreHorizontal className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" onClick={(e) => e.stopPropagation()}>
                  <DropdownMenuItem onClick={handleArchiveToggle}>
                    {notebook.archived ? (
                      <>
                        <ArchiveRestore className="h-4 w-4 mr-2" />
                        {t('notebooks.unarchive')}
                      </>
                    ) : (
                      <>
                        <Archive className="h-4 w-4 mr-2" />
                        {t('notebooks.archive')}
                      </>
                    )}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation()
                      setShowDeleteDialog(true)
                    }}
                    className="text-red-600"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    {t('common.delete')}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </CardHeader>
          
          <CardContent>
            <CardDescription className="line-clamp-2 text-sm">
              {notebook.description || t('chat.noDescription')}
            </CardDescription>

            <div className="mt-3 text-xs text-muted-foreground">
              {t('common.updated').replace('{time}', formatDistanceToNow(new Date(notebook.updated), { 
                addSuffix: true,
                locale: getDateLocale(language)
              }))}
            </div>

            {/* Item counts footer */}
            <div className="mt-3 flex items-center gap-1.5 border-t pt-3">
              <Badge variant="outline" className="text-xs flex items-center gap-1 px-1.5 py-0.5 text-primary border-primary/50">
                <FileText className="h-3 w-3" />
                <span>{notebook.source_count}</span>
              </Badge>
              <Badge variant="outline" className="text-xs flex items-center gap-1 px-1.5 py-0.5 text-primary border-primary/50">
                <StickyNote className="h-3 w-3" />
                <span>{notebook.note_count}</span>
              </Badge>
            </div>
          </CardContent>
      </Card>

      <NotebookDeleteDialog
        open={showDeleteDialog}
        onOpenChange={setShowDeleteDialog}
        notebookId={notebook.id}
        notebookName={notebook.name}
      />
    </>
  )
}
