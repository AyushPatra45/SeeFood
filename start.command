#!/bin/zsh
# Double-click on macOS to keep SeeFood running in your own Terminal window.
set -e
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Install Python 3.12, then run this launcher again."
    exit 1
  fi
  python3.12 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi
echo "SeeFood: http://127.0.0.1:8000"
echo "Keep this window open while using SeeFood. Press Ctrl+C to stop."
exec .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
