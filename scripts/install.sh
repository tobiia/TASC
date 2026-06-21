#!/bin/bash
set -e

TASC_DIR="$(cd "$(dirname "$0")/../" && pwd)"
cd "$TASC_DIR"

python3.12 -m scripts.install

echo
echo "Press Enter to close..."
read