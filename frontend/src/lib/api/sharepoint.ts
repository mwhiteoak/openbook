import { apiClient } from './client'

export interface SharePointConnection {
  connected: boolean
  site_url?: string
  folder_path?: string
  user_id?: string
}

export interface SharePointFile {
  id: string
  name: string
  path: string
  size: number
  modified: string
  file_type: string
}

export interface SharePointFolder {
  id: string
  name: string
  path: string
  children: (SharePointFile | SharePointFolder)[]
}

export interface BrowseResult {
  path: string
  items: SharePointFolder[]
  placeholder: boolean
  note: string
}

export interface ImportRequest {
  file_ids: string[]
  notebook_id: string
}

export const sharepointApi = {
  getConnection: async (): Promise<SharePointConnection> => {
    const response = await apiClient.get<SharePointConnection>('/sharepoint/connection')
    return response.data
  },

  connect: async (site_url: string, folder_path: string): Promise<SharePointConnection> => {
    const response = await apiClient.post<SharePointConnection>('/sharepoint/connect', {
      site_url,
      folder_path,
    })
    return response.data
  },

  disconnect: async (): Promise<void> => {
    await apiClient.delete('/sharepoint/connection')
  },

  browse: async (path = '/'): Promise<BrowseResult> => {
    const response = await apiClient.get<BrowseResult>('/sharepoint/browse', {
      params: { path },
    })
    return response.data
  },

  importFiles: async (data: ImportRequest) => {
    const response = await apiClient.post('/sharepoint/import', data)
    return response.data
  },
}
