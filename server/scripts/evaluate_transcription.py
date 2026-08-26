#!/usr/bin/env python3
from pathlib import Path
import sys

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app.transcription_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
