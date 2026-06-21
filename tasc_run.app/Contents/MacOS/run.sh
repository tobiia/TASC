#!/bin/bash
set -e

TASC_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$TASC_DIR"

.venv/bin/python -m src.cli run

echo
echo "Press Enter to close..."
read