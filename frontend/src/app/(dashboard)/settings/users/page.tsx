'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { useAuthStore } from '@/lib/stores/auth-store'
import { usersApi, type UserRecord } from '@/lib/api/users'
import { useTranslation } from '@/lib/hooks/use-translation'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { toast } from 'sonner'
import { UserPlus, Trash2, Shield, User } from 'lucide-react'

export default function UsersPage() {
  const { currentUser } = useAuthStore()
  const queryClient = useQueryClient()
  const { t } = useTranslation()

  const [inviteOpen, setInviteOpen] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<UserRecord | null>(null)
  const [inviteForm, setInviteForm] = useState({
    name: '',
    email: '',
    password: '',
    role: 'user',
  })

  // Redirect non-admins
  if (currentUser?.role !== 'admin') {
    return (
      <AppShell>
        <div className="flex-1 flex items-center justify-center">
          <p className="text-muted-foreground">{t('users.adminRequired')}</p>
        </div>
      </AppShell>
    )
  }

  const { data: users = [], isLoading } = useQuery({
    queryKey: ['users'],
    queryFn: usersApi.list,
  })

  const inviteMutation = useMutation({
    mutationFn: usersApi.invite,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success(t('users.inviteSuccess'))
      setInviteOpen(false)
      setInviteForm({ name: '', email: '', password: '', role: 'user' })
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || t('users.inviteFailed'))
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (userId: string) => usersApi.delete(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success(t('users.removedSuccess'))
      setDeleteTarget(null)
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || t('users.removeFailed'))
    },
  })

  const updateRoleMutation = useMutation({
    mutationFn: ({ userId, role }: { userId: string; role: string }) =>
      usersApi.update(userId, { role }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] })
      toast.success(t('users.roleUpdated'))
    },
    onError: (err: any) => {
      toast.error(err?.response?.data?.detail || t('users.roleUpdateFailed'))
    },
  })

  const handleInvite = () => {
    if (!inviteForm.name || !inviteForm.email || !inviteForm.password) {
      toast.error(t('users.fillAllFields'))
      return
    }
    inviteMutation.mutate(inviteForm)
  }

  const userCountLabel = users.length === 1
    ? t('users.count').replace('{count}', String(users.length))
    : t('users.countPlural').replace('{count}', String(users.length))

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 max-w-4xl">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl font-bold">{t('users.title')}</h1>
              <p className="text-muted-foreground text-sm mt-1">{t('users.desc')}</p>
            </div>
            <Button onClick={() => setInviteOpen(true)}>
              <UserPlus className="h-4 w-4 mr-2" />
              {t('users.inviteUser')}
            </Button>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>{t('users.teamMembers')}</CardTitle>
              <CardDescription>{userCountLabel}</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <p className="text-muted-foreground text-sm">{t('common.loading')}</p>
              ) : (
                <div className="divide-y">
                  {users.map((user) => (
                    <div key={user.id} className="flex items-center justify-between py-3">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                          {user.role === 'admin' ? (
                            <Shield className="h-4 w-4 text-primary" />
                          ) : (
                            <User className="h-4 w-4 text-muted-foreground" />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-medium">{user.name}</p>
                          <p className="text-xs text-muted-foreground">{user.email}</p>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <Select
                          value={user.role}
                          onValueChange={(role) =>
                            updateRoleMutation.mutate({ userId: user.id, role })
                          }
                          disabled={user.id === currentUser?.id}
                        >
                          <SelectTrigger className="w-28 h-8 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="user">{t('users.roleUser')}</SelectItem>
                            <SelectItem value="admin">{t('users.roleAdmin')}</SelectItem>
                          </SelectContent>
                        </Select>

                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-destructive hover:text-destructive"
                          disabled={user.id === currentUser?.id}
                          onClick={() => setDeleteTarget(user)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Invite Dialog */}
      <Dialog open={inviteOpen} onOpenChange={setInviteOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('users.inviteTitle')}</DialogTitle>
            <DialogDescription>{t('users.inviteDesc')}</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>{t('users.name')}</Label>
              <Input
                placeholder={t('users.namePlaceholder')}
                value={inviteForm.name}
                onChange={(e) => setInviteForm((f) => ({ ...f, name: e.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>{t('users.email')}</Label>
              <Input
                type="email"
                placeholder={t('users.emailPlaceholder')}
                value={inviteForm.email}
                onChange={(e) => setInviteForm((f) => ({ ...f, email: e.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>{t('users.initialPassword')}</Label>
              <Input
                type="password"
                placeholder={t('users.passwordHint')}
                value={inviteForm.password}
                onChange={(e) => setInviteForm((f) => ({ ...f, password: e.target.value }))}
              />
            </div>
            <div className="space-y-2">
              <Label>{t('users.role')}</Label>
              <Select
                value={inviteForm.role}
                onValueChange={(role) => setInviteForm((f) => ({ ...f, role }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">{t('users.roleUser')}</SelectItem>
                  <SelectItem value="admin">{t('users.roleAdmin')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteOpen(false)}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleInvite} disabled={inviteMutation.isPending}>
              {inviteMutation.isPending ? t('users.inviting') : t('users.invite')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation */}
      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('users.removeTitle')}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('users.removeDesc')
                .replace('{name}', deleteTarget?.name ?? '')
                .replace('{email}', deleteTarget?.email ?? '')}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget.id)}
              disabled={deleteMutation.isPending}
            >
              {t('users.remove')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </AppShell>
  )
}
