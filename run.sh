#!/bin/bash
set -e
cd "$(dirname "$0")"

.venv/bin/python -m src.cli run

echo
echo "Press Enter to close..."
read