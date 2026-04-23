'use client'

import { useEffect, useState } from 'react'
import { useTranslation } from '@/lib/hooks/use-translation'
import { cn } from '@/lib/utils'

/**
 * Playful loading indicator for chat streams. Shows three bouncing dots with
 * a shimmering status message that cycles every few seconds through a list of
 * domain-flavoured verbs ("Reading your sources...", "Connecting the dots..."
 * etc.) so users get a sense of progress instead of staring at a bare spinner.
 *
 * Messages are translation keys under `chat.thinking.*` in the locale files;
 * missing keys fall back to English via i18next, so adding new variants only
 * requires updating the English bundle.
 */

// Message keys cycled through while the model is thinking. Ordered loosely
// from "just started" → "still working" so the UX reads like a progression.
// Add variants here (and a matching en-US key) to expand the rotation.
const MESSAGE_KEYS = [
  'chat.thinking.reading',
  'chat.thinking.pondering',
  'chat.thinking.connecting',
  'chat.thinking.consulting',
  'chat.thinking.cross_referencing',
  'chat.thinking.brewing',
  'chat.thinking.almost',
] as const

// How long each message stays visible before the next one fades in.
const ROTATION_MS = 2800

interface ThinkingIndicatorProps {
  // When true, renders a compact inline variant (just dots, no message) for
  // use inside existing AI bubbles where the message already has context.
  compact?: boolean
  className?: string
}

export function ThinkingIndicator({ compact = false, className }: ThinkingIndicatorProps) {
  const { t } = useTranslation()
  const [messageIndex, setMessageIndex] = useState(0)

  // Pick a random starting index so reopening chat doesn't always show
  // "Reading..." first — keeps repeated interactions feeling fresh.
  useEffect(() => {
    setMessageIndex(Math.floor(Math.random() * MESSAGE_KEYS.length))
  }, [])

  // Rotate through messages while mounted. The interval only advances the
  // index — fade-in/out is handled purely by CSS via the `key` on the span
  // below, which forces React to unmount the old text and mount the new one.
  useEffect(() => {
    if (compact) return
    const interval = setInterval(() => {
      setMessageIndex((i) => (i + 1) % MESSAGE_KEYS.length)
    }, ROTATION_MS)
    return () => clearInterval(interval)
  }, [compact])

  const dots = (
    <div className={cn('flex items-center gap-1', className)} aria-hidden="true">
      <span
        className="h-1.5 w-1.5 rounded-full bg-current opacity-60 animate-bounce motion-reduce:animate-none"
        style={{ animationDelay: '0ms', animationDuration: '1.1s' }}
      />
      <span
        className="h-1.5 w-1.5 rounded-full bg-current opacity-60 animate-bounce motion-reduce:animate-none"
        style={{ animationDelay: '180ms', animationDuration: '1.1s' }}
      />
      <span
        className="h-1.5 w-1.5 rounded-full bg-current opacity-60 animate-bounce motion-reduce:animate-none"
        style={{ animationDelay: '360ms', animationDuration: '1.1s' }}
      />
    </div>
  )

  if (compact) {
    return dots
  }

  const messageKey = MESSAGE_KEYS[messageIndex]

  return (
    <div
      className="flex items-center gap-2.5"
      role="status"
      aria-live="polite"
      aria-label={t(messageKey)}
    >
      {dots}
      {/* Keyed span → React remounts on each index change, which restarts the
          fade-in keyframe so the message animates in smoothly. The shimmer
          gradient gives a subtle "processing" vibe without being distracting. */}
      <span
        key={messageKey}
        className="text-sm font-medium bg-gradient-to-r from-muted-foreground via-foreground to-muted-foreground bg-[length:200%_100%] bg-clip-text text-transparent animate-thinking-shimmer motion-reduce:animate-none motion-reduce:text-muted-foreground"
      >
        {t(messageKey)}
      </span>
    </div>
  )
}
