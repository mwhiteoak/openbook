import { enUS } from './en-US';

export const resources = {
  'en-US': { translation: enUS },
} as const;

export type TranslationKeys = typeof enUS;

export type LanguageCode = 'en-US';

export type Language = {
  code: LanguageCode;
  label: string;
};

export const languages: Language[] = [
  { code: 'en-US', label: 'English' },
];

export { enUS };
