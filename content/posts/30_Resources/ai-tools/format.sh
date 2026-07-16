#!/usr/bin/env bash
# 将 LaTeX 数学定界符替换为 KaTeX / Markdown 风格：
#   \[ ... \]  →  $$ ... $$
#   \( ... \)  →  $ ... $
#
# 用法: ./format.sh <file> [file...]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <file> [file...]" >&2
  exit 1
fi

for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "Skip (not a file): $f" >&2
    continue
  fi
  # macOS / BSD sed: -i '' 原地编辑，无备份
  sed -i '' \
    -e 's/\\\[/$$/g' \
    -e 's/\\\]/$$/g' \
    -e 's/\\(/$/g' \
    -e 's/\\)/$/g' \
    "$f"
  echo "Formatted: $f"
done
