'use client'

import { useMemo } from 'react'
import Link from 'next/link'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { ShieldAlert, AlertTriangle, ArrowRight, ExternalLink, Sparkles } from 'lucide-react'
import { useTranslation } from '@/lib/hooks/use-translation'
import { useCredentialStatus, useEnvStatus } from '@/lib/hooks/use-credentials'
import { useModels } from '@/lib/hooks/use-models'

export function SetupBanner() {
  const { t } = useTranslation()
  const { data: credentialStatus } = useCredentialStatus()
  const { data: envStatus } = useEnvStatus()
  const { data: models } = useModels()

  const encryptionReady = credentialStatus?.encryption_configured ?? true

  // Show "no model" banner only after models have loaded and none are language-type
  const hasLanguageModel = models === undefined || models.some((m: { type: string }) => m.type === 'language')
  const showNoModelBanner = encryptionReady && models !== undefined && !hasLanguageModel

  const providersToMigrate = useMemo(() => {
    if (!envStatus || !credentialStatus) return []
    const providers: string[] = []
    for (const provider in envStatus) {
      if (envStatus[provider] && credentialStatus.source[provider] === 'environment') {
        providers.push(provider)
      }
    }
    return providers
  }, [envStatus, credentialStatus])

  const banners: React.ReactNode[] = []

  if (!encryptionReady) {
    banners.push(
      <Alert key="encryption" className="border-red-500/50 bg-red-50 dark:bg-red-950/20">
        <ShieldAlert className="h-4 w-4 text-red-600 dark:text-red-400" />
        <AlertTitle className="text-red-800 dark:text-red-200">
          {t('setupBanner.encryptionRequired')}
        </AlertTitle>
        <AlertDescription className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between text-red-700 dark:text-red-300">
          <span>{t('setupBanner.encryptionRequiredDescription')}</span>
          <a
            href="https://github.com/lfnovo/open-notebook/blob/main/docs/3-USER-GUIDE/api-configuration.md#encryption-setup"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center shrink-0 text-sm font-medium underline underline-offset-2 hover:text-red-900 dark:hover:text-red-100"
          >
            {t('setupBanner.viewDocs')}
            <ExternalLink className="ml-1 h-3 w-3" />
          </a>
        </AlertDescription>
      </Alert>
    )
  }

  if (showNoModelBanner) {
    banners.push(
      <Alert key="no-model" className="border-blue-500/50 bg-blue-50 dark:bg-blue-950/20">
        <Sparkles className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <AlertTitle className="text-blue-800 dark:text-blue-200">
          {t('setupBanner.noModelConfigured')}
        </AlertTitle>
        <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-blue-700 dark:text-blue-300">
            {t('setupBanner.noModelConfiguredDescription')}
          </span>
          <Button
            variant="outline"
            size="sm"
            asChild
            className="shrink-0 border-blue-500 text-blue-700 hover:bg-blue-100 dark:border-blue-400 dark:text-blue-300 dark:hover:bg-blue-900/30"
          >
            <Link href="/settings/api-keys">
              {t('setupBanner.goToSettings')}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  if (providersToMigrate.length > 0) {
    banners.push(
      <Alert key="migration" className="border-amber-500/50 bg-amber-50 dark:bg-amber-950/20">
        <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
        <AlertTitle className="text-amber-800 dark:text-amber-200">
          {t('setupBanner.migrationAvailable')}
        </AlertTitle>
        <AlertDescription className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-amber-700 dark:text-amber-300">
            {t('setupBanner.migrationDescription').replace('{count}', providersToMigrate.length.toString())}
          </span>
          <Button
            variant="outline"
            size="sm"
            asChild
            className="shrink-0 border-amber-500 text-amber-700 hover:bg-amber-100 dark:border-amber-400 dark:text-amber-300 dark:hover:bg-amber-900/30"
          >
            <Link href="/settings/api-keys">
              {t('setupBanner.goToSettings')}
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  if (banners.length === 0) return null

  return (
    <div className="px-4 pt-3 flex flex-col gap-2">
      {banners}
    </div>
  )
}
