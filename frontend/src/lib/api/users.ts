import { apiClient } from './client'

export interface UserRecord {
  id: string
  email: string
  name: string
  role: string
  created?: string
  updated?: string
}

export interface InviteUserRequest {
  email: string
  name: string
  password: string
  role: string
}

export interface UpdateUserRequest {
  name?: string
  role?: string
  password?: string
}

export const usersApi = {
  list: async (): Promise<UserRecord[]> => {
    const response = await apiClient.get<UserRecord[]>('/users')
    return response.data
  },

  me: async (): Promise<UserRecord> => {
    const response = await apiClient.get<UserRecord>('/users/me')
    return response.data
  },

  invite: async (data: InviteUserRequest): Promise<UserRecord> => {
    const response = await apiClient.post<UserRecord>('/auth/invite', data)
    return response.data
  },

  update: async (userId: string, data: UpdateUserRequest): Promise<UserRecord> => {
    const response = await apiClient.put<UserRecord>(`/users/${userId}`, data)
    return response.data
  },

  updateMe: async (data: UpdateUserRequest): Promise<UserRecord> => {
    const response = await apiClient.put<UserRecord>('/users/me', data)
    return response.data
  },

  delete: async (userId: string): Promise<void> => {
    await apiClient.delete(`/users/${userId}`)
  },
}
