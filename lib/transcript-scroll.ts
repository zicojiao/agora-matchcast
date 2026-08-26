type TranscriptScrollRail = {
  scrollHeight: number;
  scrollTop: number;
};

export function scrollTranscriptToLatest(rail: TranscriptScrollRail) {
  rail.scrollTop = rail.scrollHeight;
}
