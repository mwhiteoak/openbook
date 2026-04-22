'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { sharepointApi, type SharePointFile } from '@/lib/api/sharepoint'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import { useTranslation } from '@/lib/hooks/use-translation'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { toast } from 'sonner'
import {
  FolderOpen,
  FileText,
  Link2,
  Link2Off,
  RefreshCw,
  Download,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react'

function fileIcon(_fileType: string) {
  return <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function SharePointPage() {
  const queryClient = useQueryClient()
  const { t } = useTranslation()
  const [siteUrl, setSiteUrl] = useState('')
  const [folderPath, setFolderPath] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [targetNotebook, setTargetNotebook] = useState('')

  const { data: connection, isLoading: connLoading } = useQuery({
    queryKey: ['sharepoint-connection'],
    queryFn: sharepointApi.getConnection,
  })

  const { data: browseResult, isLoading: browseLoading, refetch: refetchBrowse } = useQuery({
    queryKey: ['sharepoint-browse'],
    queryFn: () => sharepointApi.browse('/'),
    enabled: connection?.connected === true,
  })

  const { data: notebooks } = useNotebooks()

  const connectMutation = useMutation({
    mutationFn: () => sharepointApi.connect(siteUrl, folderPath),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sharepoint-connection'] })
      queryClient.invalidateQueries({ queryKey: ['sharepoint-browse'] })
      toast.success(t('sharepoint.connectSuccess'))
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || t('sharepoint.connectFailed'))
    },
  })

  const disconnectMutation = useMutation({
    mutationFn: sharepointApi.disconnect,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sharepoint-connection'] })
      queryClient.invalidateQueries({ queryKey: ['sharepoint-browse'] })
      setSelectedFiles(new Set())
      toast.success(t('sharepoint.disconnectSuccess'))
    },
  })

  const importMutation = useMutation({
    mutationFn: () =>
      sharepointApi.importFiles({
        file_ids: Array.from(selectedFiles),
        notebook_id: targetNotebook,
      }),
    onSuccess: () => {
      toast.success(t('sharepoint.importQueued').replace('{count}', String(selectedFiles.size)))
      setSelectedFiles(new Set())
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || t('sharepoint.importFailed'))
    },
  })

  const toggleFile = (fileId: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      if (next.has(fileId)) next.delete(fileId)
      else next.add(fileId)
      return next
    })
  }

  if (connLoading) {
    return (
      <AppShell>
        <div className="flex-1 flex items-center justify-center">
          <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      </AppShell>
    )
  }

  const selectedLabel = selectedFiles.size === 1
    ? t('sharepoint.filesSelected').replace('{count}', String(selectedFiles.size))
    : t('sharepoint.filesSelectedPlural').replace('{count}', String(selectedFiles.size))

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 max-w-4xl space-y-6">
          <div>
            <h1 className="text-2xl font-bold">{t('sharepoint.title')}</h1>
            <p className="text-muted-foreground text-sm mt-1">{t('sharepoint.desc')}</p>
          </div>

          {/* Placeholder notice */}
          <div className="flex items-start gap-2 p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-md text-sm text-amber-800 dark:text-amber-200">
            <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
            <span>
              <strong>MVP placeholder:</strong> {t('sharepoint.mvpNotice')}
            </span>
          </div>

          {/* Connection Card */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    {connection?.connected ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500" />
                    ) : (
                      <Link2Off className="h-5 w-5 text-muted-foreground" />
                    )}
                    {t('sharepoint.connectionTitle')}
                  </CardTitle>
                  <CardDescription>
                    {connection?.connected
                      ? t('sharepoint.connectedTo').replace('{siteUrl}', connection.site_url)
                      : t('sharepoint.notConnected')}
                  </CardDescription>
                </div>
                {connection?.connected && (
                  <Badge variant="outline" className="text-green-600 border-green-300">
                    {t('sharepoint.connected')}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {connection?.connected ? (
                <div className="space-y-3">
                  <div className="text-sm space-y-1">
                    <p>
                      <span className="text-muted-foreground">{t('sharepoint.siteUrlInfo')}: </span>
                      <span className="font-mono">{connection.site_url}</span>
                    </p>
                    <p>
                      <span className="text-muted-foreground">{t('sharepoint.folderInfo')}: </span>
                      <span className="font-mono">{connection.folder_path}</span>
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => disconnectMutation.mutate()}
                    disabled={disconnectMutation.isPending}
                  >
                    <Link2Off className="h-4 w-4 mr-2" />
                    {t('sharepoint.disconnect')}
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>{t('sharepoint.siteUrlLabel')}</Label>
                    <Input
                      placeholder={t('sharepoint.siteUrlPlaceholder')}
                      value={siteUrl}
                      onChange={(e) => setSiteUrl(e.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>{t('sharepoint.folderPathLabel')}</Label>
                    <Input
                      placeholder={t('sharepoint.folderPathPlaceholder')}
                      value={folderPath}
                      onChange={(e) => setFolderPath(e.target.value)}
                    />
                  </div>
                  <Button
                    onClick={() => connectMutation.mutate()}
                    disabled={connectMutation.isPending || !siteUrl || !folderPath}
                  >
                    <Link2 className="h-4 w-4 mr-2" />
                    {connectMutation.isPending ? t('sharepoint.connecting') : t('sharepoint.connect')}
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* File Browser */}
          {connection?.connected && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{t('sharepoint.browseTitle')}</CardTitle>
                    <CardDescription>{t('sharepoint.browseDesc')}</CardDescription>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => refetchBrowse()}
                    disabled={browseLoading}
                  >
                    <RefreshCw className={`h-4 w-4 mr-2 ${browseLoading ? 'animate-spin' : ''}`} />
                    {t('sharepoint.refresh')}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {browseLoading ? (
                  <p className="text-sm text-muted-foreground">{t('sharepoint.loadingFiles')}</p>
                ) : (
                  <>
                    {(browseResult?.items ?? []).map((folder) => (
                      <div key={folder.id}>
                        <div className="flex items-center gap-2 mb-2 text-sm font-medium text-muted-foreground">
                          <FolderOpen className="h-4 w-4" />
                          {folder.name}
                        </div>
                        <div className="ml-6 space-y-1">
                          {(folder.children ?? [])
                            .filter((c): c is SharePointFile => 'size' in c)
                            .map((file) => (
                              <label
                                key={file.id}
                                className="flex items-center gap-3 p-2 rounded hover:bg-muted cursor-pointer"
                              >
                                <Checkbox
                                  checked={selectedFiles.has(file.id)}
                                  onCheckedChange={() => toggleFile(file.id)}
                                />
                                {fileIcon(file.file_type)}
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm truncate">{file.name}</p>
                                  <p className="text-xs text-muted-foreground">
                                    {formatBytes(file.size)} ·{' '}
                                    {new Date(file.modified).toLocaleDateString()}
                                  </p>
                                </div>
                                <Badge variant="outline" className="text-xs uppercase">
                                  {file.file_type}
                                </Badge>
                              </label>
                            ))}
                        </div>
                      </div>
                    ))}

                    {/* Import bar */}
                    {selectedFiles.size > 0 && (
                      <div className="flex items-center gap-3 pt-4 border-t">
                        <span className="text-sm text-muted-foreground">{selectedLabel}</span>
                        <Select value={targetNotebook} onValueChange={setTargetNotebook}>
                          <SelectTrigger className="flex-1">
                            <SelectValue placeholder={t('sharepoint.selectNotebook')} />
                          </SelectTrigger>
                          <SelectContent>
                            {(notebooks ?? []).map((nb: any) => (
                              <SelectItem key={nb.id} value={nb.id}>
                                {nb.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          onClick={() => importMutation.mutate()}
                          disabled={importMutation.isPending || !targetNotebook}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          {importMutation.isPending ? t('sharepoint.importing') : t('sharepoint.import')}
                        </Button>
                      </div>
                    )}
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </AppShell>
  )
}
