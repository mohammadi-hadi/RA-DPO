#!/usr/bin/env python3
"""Run the experiment pipeline (thin wrapper around ra_dpo.cli).

Kept so the documented ``python scripts/run_pipeline.py ...`` commands keep
working from a plain clone without installing the package. See
``ra-dpo --help`` (or ``ra_dpo/cli.py``) for all options.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ra_dpo.cli import main

if __name__ == "__main__":
    main()
