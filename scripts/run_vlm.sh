#!/usr/bin/env bash
# Companion VLM process. Keep this separate from the Java stitcher.
set -euo pipefail
cd "$(dirname "$0")/../vlm"
python3 -m pip install -q -r requirements.txt
export VLM_BACKEND="${VLM_BACKEND:-hybrid}"
export VLM_HOST="${VLM_HOST:-127.0.0.1}"
export VLM_PORT="${VLM_PORT:-8088}"
exec python3 server.py
