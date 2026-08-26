import MatchCastDemo from '@/components/MatchCastDemo';
import {
  availableTranscribers,
  isPublicDemoMode,
} from '@/lib/transcribers';

export const dynamic = 'force-dynamic';

export default async function Home() {
  return (
    <MatchCastDemo
      transcribers={availableTranscribers(isPublicDemoMode())}
    />
  );
}
