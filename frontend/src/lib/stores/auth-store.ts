'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { getApiUrl } from '@/lib/config'

export interface CurrentUser {
  id: string
  email: string
  name: string
  role: string
}

interface AuthState {
  isAuthenticated: boolean
  token: string | null
  currentUser: CurrentUser | null
  isLoading: boolean
  error: string | null
  lastAuthCheck: number | null
  isCheckingAuth: boolean
  hasHydrated: boolean
  authRequired: boolean | null
  setHasHydrated: (state: boolean) => void
  checkAuthRequired: () => Promise<boolean>
  login: (email: string, password: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<boolean>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      token: null,
      currentUser: null,
      isLoading: false,
      error: null,
      lastAuthCheck: null,
      isCheckingAuth: false,
      hasHydrated: false,
      authRequired: null,

      setHasHydrated: (state: boolean) => {
        set({ hasHydrated: state })
      },

      checkAuthRequired: async () => {
        try {
          const apiUrl = await getApiUrl()
          const response = await fetch(`${apiUrl}/api/auth/status`, {
            cache: 'no-store',
          })

          if (!response.ok) {
            throw new Error(`Auth status check failed: ${response.status}`)
          }

          const data = await response.json()
          // In multi-user mode, auth is always required
          const required = data.auth_enabled !== false
          set({ authRequired: required })

          return required
        } catch (error) {
          console.error('Failed to check auth status:', error)

          if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            set({
              error: 'Unable to connect to server. Please check if the API is running.',
              authRequired: null,
            })
          } else {
            set({ authRequired: true })
          }

          throw error
        }
      },

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const apiUrl = await getApiUrl()

          const response = await fetch(`${apiUrl}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
          })

          if (response.ok) {
            const data = await response.json()
            set({
              isAuthenticated: true,
              token: data.access_token,
              currentUser: data.user as CurrentUser,
              isLoading: false,
              lastAuthCheck: Date.now(),
              error: null,
              authRequired: true,
            })
            return true
          } else {
            let errorMessage = 'Authentication failed'
            if (response.status === 401) {
              errorMessage = 'Invalid email or password. Please try again.'
            } else if (response.status >= 500) {
              errorMessage = 'Server error. Please try again later.'
            } else {
              try {
                const body = await response.json()
                errorMessage = body.detail || errorMessage
              } catch {}
            }

            set({ error: errorMessage, isLoading: false, isAuthenticated: false, token: null, currentUser: null })
            return false
          }
        } catch (error) {
          console.error('Network error during auth:', error)
          const errorMessage =
            error instanceof TypeError && error.message.includes('Failed to fetch')
              ? 'Unable to connect to server. Please check if the API is running.'
              : 'An unexpected error occurred during authentication'

          set({ error: errorMessage, isLoading: false, isAuthenticated: false, token: null, currentUser: null })
          return false
        }
      },

      logout: () => {
        set({ isAuthenticated: false, token: null, currentUser: null, error: null })
      },

      checkAuth: async () => {
        const state = get()
        const { token, lastAuthCheck, isCheckingAuth, isAuthenticated } = state

        if (isCheckingAuth) return isAuthenticated
        if (!token) return false

        const now = Date.now()
        if (isAuthenticated && lastAuthCheck && now - lastAuthCheck < 30000) {
          return true
        }

        set({ isCheckingAuth: true })

        try {
          const apiUrl = await getApiUrl()
          const response = await fetch(`${apiUrl}/api/auth/me`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          })

          if (response.ok) {
            const user = await response.json()
            set({
              isAuthenticated: true,
              currentUser: user as CurrentUser,
              lastAuthCheck: now,
              isCheckingAuth: false,
            })
            return true
          } else {
            set({ isAuthenticated: false, token: null, currentUser: null, lastAuthCheck: null, isCheckingAuth: false })
            return false
          }
        } catch (error) {
          console.error('checkAuth error:', error)
          set({ isAuthenticated: false, token: null, currentUser: null, lastAuthCheck: null, isCheckingAuth: false })
          return false
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        isAuthenticated: state.isAuthenticated,
        currentUser: state.currentUser,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true)
      },
    }
  )
)
