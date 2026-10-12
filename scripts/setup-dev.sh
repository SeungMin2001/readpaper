#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm_config_cache=/tmp/readpaper-npm-cache npm ci --prefix frontend --no-audit --no-fund
npm run build --prefix frontend
