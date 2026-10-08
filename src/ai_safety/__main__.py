"""Entrypoint for executing ai_safety module directly with python -m ai_safety."""

import sys
from ai_safety.cli import main

if __name__ == "__main__":
    sys.exit(main())
