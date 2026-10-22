#!/usr/bin/env bash
# Bring back the per-epoch loss histories (losshist_*.npz) of the large sweeps.
# They are kept out of this repository (about 4 GB) and live, at the same paths, in GubioGL/CBPF-Geral.
#
#   bash data/fetch_full_histories.sh            # run from the repository root
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDERS=(data/Inverso/classical_inverse data/Inverso/quantum_inverse)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/GubioGL/CBPF-Geral "$TMP/src"
git -C "$TMP/src" sparse-checkout set "${FOLDERS[@]}"
git -C "$TMP/src" checkout
for f in "${FOLDERS[@]}"; do
  mkdir -p "$f"
  (cd "$TMP/src/$f" && find . -name 'losshist_*.npz' -print0 | tar cf - --null -T -) | (cd "$f" && tar xf -)
  echo "$f: $(find "$f" -name 'losshist_*.npz' | wc -l) histories"
done
