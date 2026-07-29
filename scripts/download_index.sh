#!/usr/bin/env bash
# Download and unpack the prebuilt keyframe indexes (README §5.2).
#
# These are the embeddings the official submission was produced with. Rebuilding them
# from video takes ~8 GPU-hours (extract_embed.py + extract_embed_qwen.py), so they are
# shipped as release assets instead.
#
# index_qwen8b.tar is 4.8 GB, above the 2 GB per-file limit on GitHub Releases, so it
# is uploaded in parts and reassembled here.
#
# Usage:  bash scripts/download_index.sh [target_dir]      (default: repo root)
set -euo pipefail

# Release assets — FILL IN before submitting (or export TEMPORUN_INDEX_URL).
BASE_URL="${TEMPORUN_INDEX_URL:-https://github.com/nguyenminh2007tuan-star/TempoRun_minhnht/releases/download/v1.0}"
FILES=(
    index_siglip2.tar
    index_qwen8b.tar.part00
    index_qwen8b.tar.part01
    index_qwen8b.tar.part02
    index.sha256
)

TARGET="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$TARGET"
echo "[download] target: $TARGET"

for f in "${FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "[skip] $f already present"
    else
        echo "[get ] $f"
        curl -fL --retry 3 -o "$f" "$BASE_URL/$f"
    fi
done

echo "[verify] sha256 of downloaded parts"
sha256sum -c index.sha256

echo "[join ] index_qwen8b.tar.part* -> index_qwen8b.tar"
cat index_qwen8b.tar.part* > index_qwen8b.tar

for f in index_siglip2.tar index_qwen8b.tar; do
    echo "[untar] $f"
    tar xf "$f"
done

echo "[check] shard counts (expected: qwen8b 5006, siglip2 5004 — see README §11)"
echo "  index_qwen8b : $(ls index_qwen8b/shards/*.npz 2>/dev/null | wc -l)"
echo "  index_siglip2: $(ls index_siglip2/shards/*.npz 2>/dev/null | wc -l)"
echo "[done] you can now delete the .tar / .part files"
