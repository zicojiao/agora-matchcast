from collections.abc import Iterable


# Curated for the English MatchCast highlight reel. Keep this list small and
# limited to distinctive names or phrases that generic speech recognition is
# likely to split or misspell. Ordinary English and generic gameplay words
# create harmful bias and do not belong here.
MATCHCAST_CLIP_VOCABULARY: tuple[str, ...] = (
    # Distinctive map, objective, and item names.
    "Summoner's Rift",
    "Baron Nashor",
    "Zhonya's Hourglass",
    # Teams heard across the highlight reel.
    "SK Telecom T1",
    "T1",
    "SKT",
    "ROX Tigers",
    "JD Gaming",
    "JDG",
    "G2 Esports",
    "G2",
    "Royal Never Give Up",
    "RNG",
    "Weibo Gaming",
    "CLG",
    "TSM",
    # Players whose spellings benefit from an explicit hint.
    "Faker",
    "Huni",
    "Perkz",
    "Rekkles",
    "Gumayusi",
    "Keria",
    "Oner",
    "Zeus",
    "Ryu",
    # Champions and distinctive abilities heard in the reel.
    "Orianna",
    "Command: Shockwave",
    "Shockwave",
    "Ekko",
    "Chronobreak",
    "Caitlyn",
    "Caliber Net",
    "Lulu",
    "Wild Growth",
    "Glitterlance",
    "Gangplank",
    "Cannon Barrage",
    "Zilean",
    "Olaf",
    "Teleport",
    "Zed",
    "Death Mark",
    "Quicksilver Sash",
    "QSS",
)


# Dedicated Gemini Transcribe can over-bias on long custom vocabulary lists.
# Keep its default vocabulary compact and independent from the broader preset
# used by the transcription adapter.
GEMINI_TRANSCRIBE_VOCABULARY: tuple[str, ...] = (
    "T1",
    "Faker",
    "Huni",
    "Rekkles",
    "Keria",
    "Ryu",
    "Orianna",
    "Shockwave",
    "Death Mark",
    "QSS",
)


def merge_vocabulary(*groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            term = item.strip()
            normalized = term.casefold()
            if not term or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(term)
    return merged
