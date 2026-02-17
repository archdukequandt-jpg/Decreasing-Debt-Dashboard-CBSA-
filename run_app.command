#!/bin/bash
set -e
# If launched via Finder, ensure we run from this script's directory
cd "$(dirname "$0")"
set -e

cd "$(dirname "$0")"

chmod +x "$0" 2>/dev/null || true
xattr -d com.apple.quarantine "$0" 2>/dev/null || true

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

streamlit run app.py
