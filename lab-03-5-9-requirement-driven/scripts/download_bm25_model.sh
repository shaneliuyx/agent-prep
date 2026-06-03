#!/usr/bin/env bash
# Pre-fetch the Qdrant/bm25 sparse model for fastembed (mem0 hybrid dense+BM25).
#
# WHY: fastembed pulls Qdrant/bm25 through HuggingFace's Xet CDN, which fails on
# some networks (and in China). The files are tiny plain-text stopword lists, so
# curl over plain HTTPS works where Xet does not. This script lays them out in
# the HuggingFace hub cache structure (blobs + refs + snapshot symlinks) so
# fastembed loads them OFFLINE — no Xet, no download storm.
#
# Usage:
#   chmod +x scripts/download_bm25_model.sh
#   ./scripts/download_bm25_model.sh
#   # China / Xet-blocked networks — use a plain-LFS mirror (NOT for Xet repos):
#   HF_ENDPOINT=https://hf-mirror.com ./scripts/download_bm25_model.sh
#
# After running, set in .env so mem0's internal SparseTextEmbedding finds it:
#   FASTEMBED_CACHE_PATH=<absolute path to ~/.cache/fastembed>
# (mem0 calls SparseTextEmbedding("Qdrant/bm25") with no cache_dir, so the env
#  is the only way to steer it; ~ is NOT expanded — use an absolute path.)

set -euo pipefail

# Pinned commit of Qdrant/bm25. Update if the upstream repo revs (check
# https://huggingface.co/Qdrant/bm25/commits/main).
HASH="e499a1f8d6bec960aab5533a0941bf914e70faf9"
CACHE_DIR="${FASTEMBED_CACHE_PATH:-$HOME/.cache/fastembed}"
BASE="$CACHE_DIR/models--Qdrant--bm25"
DEST="$BASE/snapshots/$HASH"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_BASE="$HF_ENDPOINT/Qdrant/bm25/resolve/main"

echo "Cache dir : $CACHE_DIR"
echo "Endpoint  : $HF_ENDPOINT"
echo "Creating cache dirs..."
mkdir -p "$DEST" "$BASE/blobs" "$BASE/refs"
echo -n "$HASH" > "$BASE/refs/main"

# mock.file is a placeholder fastembed's loader expects — not a real HF file.
echo -n "mock" > "$DEST/mock.file"

FILES=(
  arabic.txt danish.txt dutch.txt english.txt finnish.txt french.txt
  german.txt hungarian.txt italian.txt norwegian.txt portuguese.txt
  romanian.txt russian.txt spanish.txt swedish.txt turkish.txt
)

echo "Downloading ${#FILES[@]} stopword files..."
for f in "${FILES[@]}"; do
  printf "  %-20s" "$f"
  rm -f "$DEST/$f"   # drop any existing symlink so curl writes a fresh plain file (re-run safe)
  if curl -fsSL "$HF_BASE/$f" -o "$DEST/$f" 2>/dev/null; then
    echo "ok ($(wc -c < "$DEST/$f") bytes)"
  else
    echo "FAILED"
    exit 1
  fi
done

echo ""
echo "Building HuggingFace hub cache structure (blobs + symlinks)..."
# fastembed resolves the model through huggingface_hub, which expects
# snapshots/<hash>/<file> to be symlinks into blobs/<sha256>. Convert the
# plain files we just downloaded into that layout so a cache lookup succeeds.
cat > /tmp/_bm25_blobs.py <<'PYEOF'
import hashlib, shutil, sys
from pathlib import Path
snap, blobs_dir = Path(sys.argv[1]), Path(sys.argv[2])
blobs_dir.mkdir(parents=True, exist_ok=True)
for f in sorted(snap.iterdir()):
    if not f.is_file() or f.is_symlink():
        continue
    sha = "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    blob = blobs_dir / sha
    shutil.copy2(f, blob)
    f.unlink()
    f.symlink_to(Path("../../blobs") / sha)
    print(f"  {f.name} -> blobs/{sha[:16]}...")
PYEOF
# Prefer the lab's own interpreter so fastembed resolves against lab deps.
PY="$(dirname "$0")/../.venv/bin/python3"
[[ -f "$PY" ]] || PY="python3"
"$PY" /tmp/_bm25_blobs.py "$DEST" "$BASE/blobs"
rm /tmp/_bm25_blobs.py

echo ""
echo "Verifying via the lab's consumption path (SparseTextEmbedding + FASTEMBED_CACHE_PATH)..."
# This mirrors how mem0 loads BM25 internally: SparseTextEmbedding("Qdrant/bm25")
# with no cache_dir, steered by FASTEMBED_CACHE_PATH. If this prints terms, mem0
# will too.
FASTEMBED_CACHE_PATH="$CACHE_DIR" "$PY" - <<'PYEOF'
from fastembed import SparseTextEmbedding
m = SparseTextEmbedding(model_name="Qdrant/bm25")
out = list(m.embed(["return boots to Zara and pick up dry cleaning"]))
print("BM25 ok — nonzero terms:", len(out[0].indices))
PYEOF

echo ""
echo "Done. Model at: $DEST"
echo "Next: put FASTEMBED_CACHE_PATH=$CACHE_DIR in .env (absolute path),"
echo "      then clear mem0_* Qdrant collections so they recreate with the"
echo "      bm25 sparse-vector slot (mem0 adds it only at collection creation)."
