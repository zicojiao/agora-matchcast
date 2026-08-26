export const TRANSCRIBERS = [
  {
    value: 'gemini-transcribe',
    label: 'gemini-3.5-transcribe-live',
    detail: '',
    public: true,
  },
] as const;

export type TranscriberId = (typeof TRANSCRIBERS)[number]['value'];
export type TranscriberOption = (typeof TRANSCRIBERS)[number];

export function isPublicDemoMode(
  value = process.env.NEXT_PUBLIC_PUBLIC_DEMO_MODE,
) {
  return value?.trim().toLowerCase() === 'true';
}

export function availableTranscribers(
  publicDemoMode = isPublicDemoMode(),
) {
  return publicDemoMode
    ? TRANSCRIBERS.filter((transcriber) => transcriber.public)
    : TRANSCRIBERS;
}

export function isTranscriberId(value: unknown): value is TranscriberId {
  return TRANSCRIBERS.some((transcriber) => transcriber.value === value);
}

export function isAllowedTranscriberId(
  value: unknown,
  publicDemoMode = isPublicDemoMode(),
): value is TranscriberId {
  return availableTranscribers(publicDemoMode).some(
    (transcriber) => transcriber.value === value,
  );
}
