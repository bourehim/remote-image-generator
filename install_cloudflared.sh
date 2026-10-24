#!/usr/bin/env bash
# Download the pinned cloudflared binary for linux-amd64 into ./bin (or $1).
# launch.py does this automatically; this script is for manual/offline use.
set -euo pipefail

VERSION="${RIMAGE_CLOUDFLARED_VERSION:-2026.10.3}"
DEST_DIR="${1:-bin}"
URL="https://github.com/cloudflare/cloudflared/releases/download/${VERSION}/cloudflared-linux-amd64"
TARGET="$DEST_DIR/cloudflared"

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
mkdir -p "$DEST_DIR"

echo "downloading cloudflared ${VERSION} -> ${TARGET}"
curl -fsSL --retry 3 --connect-timeout 15 -o "${TARGET}.part" "$URL"
if [ -n "${RIMAGE_CLOUDFLARED_SHA256:-}" ]; then
  echo "${RIMAGE_CLOUDFLARED_SHA256}  ${TARGET}.part" | sha256sum -c -
else
  echo "warning: RIMAGE_CLOUDFLARED_SHA256 not set; skipping checksum verification" >&2
fi
chmod 0755 "${TARGET}.part"
mv "${TARGET}.part" "$TARGET"
"$TARGET" --version
echo "ok: $TARGET"
