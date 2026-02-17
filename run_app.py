"""run_app.py

Standalone launcher helpers for the Streamlit app.

Why this exists:
- On some systems, double-clicking a *.command file may fail due to permissions.
- This Python launcher works cross-platform and does not require the `streamlit` command to be on PATH.

Usage:
  python run_app.py

Programmatic usage:
  from run_app import command, run
  print(command())
  run()

Credits:
  Creators: Ryan Childs (ryanchilds10@gmail.com) · James Quandt (archdukequandt@gmail.com) · James Belhund (jamesbelhund@gmail.com)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def command() -> str:
    """Return the shell command that launches the Streamlit app."""
    app_path = Path(__file__).resolve().parent / "app.py"
    # Use the current Python interpreter to ensure we use the same environment
    return f'"{sys.executable}" -m streamlit run "{app_path}"'


def run() -> int:
    """Launch the app via subprocess. Returns the process exit code."""
    app_path = Path(__file__).resolve().parent / "app.py"
    # Prefer module invocation to avoid PATH issues
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    raise SystemExit(run())
