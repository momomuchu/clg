#!/bin/sh
# Symlink clg tools into ~/.local/bin (repo stays the source of truth).
set -eu
REPO="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/.local/bin"
for tool in clg clg-account clg-fleet; do
    ln -sf "$REPO/bin/$tool" "$HOME/.local/bin/$tool"
    echo "✓ $HOME/.local/bin/$tool -> $REPO/bin/$tool"
done
echo "Done. Make sure ~/.local/bin is on your PATH."
