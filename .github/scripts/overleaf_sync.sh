#!/usr/bin/env bash
# Two-way sync between paper/ in this repository and an Overleaf project (Overleaf Git integration).
#
# Overleaf edits are merged into main with a real three-way merge (base = the last synced commit),
# then paper/ is pushed to Overleaf. A conflicting edit on both sides stops the run and leaves the
# Overleaf side on the branch `overleaf-conflict` for a manual merge.
#
# Environment
#   OVERLEAF_PROJECT_ID   id from the project URL (overleaf.com/project/<id>)
#   OVERLEAF_TOKEN        Overleaf Git authentication token (Account Settings -> Git integration)
#   OVERLEAF_URL          optional: full remote URL instead of the two above (used by the tests)
#   FIRST_SYNC            auto | github | overleaf   -- who wins the very first sync (default auto:
#                         GitHub, unless paper/ only holds the starter skeleton and Overleaf has a draft).
#                         Nothing is lost either way: Overleaf keeps every earlier version in its History.
#   PAPER_DIR             default "paper";  BRANCH  default "main"
set -euo pipefail

PAPER_DIR=${PAPER_DIR:-paper}
BRANCH=${BRANCH:-main}
FIRST_SYNC=${FIRST_SYNC:-auto}
STATE=.github/overleaf-sync-state
OL_DIR=$(mktemp -d)
trap 'rm -rf "$OL_DIR"' EXIT

if [[ -z "${OVERLEAF_URL:-}" ]]; then
  : "${OVERLEAF_PROJECT_ID:?set OVERLEAF_PROJECT_ID}"
  if [[ -z "${OVERLEAF_TOKEN:-}" ]]; then
    echo "::warning::Overleaf sync skipped: add the OVERLEAF_TOKEN secret (Settings -> Secrets and variables -> Actions)."
    exit 0
  fi
  OVERLEAF_URL="https://git:${OVERLEAF_TOKEN}@git.overleaf.com/${OVERLEAF_PROJECT_ID}"
fi

log() { echo "[overleaf-sync] $*"; }

# copy the contents of $1 into $2, replacing everything in $2 except its .git
mirror() {
  local src=$1 dst=$2
  mkdir -p "$dst"
  find "$dst" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
  (cd "$src" && tar cf - --exclude=./.git .) | (cd "$dst" && tar xf -)
}
same_content() { diff -rq --exclude=.git "$OL_DIR" "$PAPER_DIR" > /dev/null 2>&1; }

overleaf_commit_push() {   # paper/ -> Overleaf
  mirror "$PAPER_DIR" "$OL_DIR"
  git -C "$OL_DIR" add -A
  if git -C "$OL_DIR" diff --cached --quiet; then log "Overleaf already up to date"; return; fi
  git -C "$OL_DIR" commit -q -m "Update from GitHub ($(git rev-parse --short HEAD))"
  git -C "$OL_DIR" push -q origin HEAD:master
  log "pushed paper/ to Overleaf"
}

save_state() {
  mkdir -p "$(dirname "$STATE")"
  printf 'overleaf=%s\ngithub=%s\n' "$(git -C "$OL_DIR" rev-parse HEAD)" "$(git rev-parse HEAD)" > "$STATE"
  git add "$STATE"
  git commit -q -m "Overleaf sync state [skip ci]"
  git push -q origin "HEAD:$BRANCH"
}

git config user.name  "overleaf-sync[bot]"
git config user.email "overleaf-sync@users.noreply.github.com"
git clone -q "$OVERLEAF_URL" "$OL_DIR"
git -C "$OL_DIR" config user.name  "overleaf-sync[bot]"
git -C "$OL_DIR" config user.email "overleaf-sync@users.noreply.github.com"
OL_HEAD=$(git -C "$OL_DIR" rev-parse HEAD)
mkdir -p "$PAPER_DIR"

# ------------------------------------------------------------------ first sync
if [[ ! -f "$STATE" ]]; then
  if same_content; then
    log "first sync: Overleaf and $PAPER_DIR/ already match"; save_state; exit 0
  fi
  winner=$FIRST_SYNC
  if [[ $winner == auto ]]; then
    n_ol=$(find "$OL_DIR" -type f -not -path '*/.git/*' | wc -l)
    lines=$(cat "$OL_DIR"/*.tex 2>/dev/null | wc -l)
    if [[ -f "$PAPER_DIR/.placeholder" ]] && (( n_ol > 3 || lines > 80 )); then
      winner=overleaf                    # GitHub only has the starter skeleton, Overleaf has a draft
    else
      winner=github
    fi
  fi
  log "first sync: $winner wins"
  if [[ $winner == overleaf ]]; then
    if [[ -f "$PAPER_DIR/.placeholder" ]]; then
      # drop the starter files listed in .placeholder, keep everything else (e.g. generated figures),
      # and lay the Overleaf project on top
      while read -r f; do [[ -n $f ]] && rm -f "$PAPER_DIR/$f"; done < "$PAPER_DIR/.placeholder"
      rm -f "$PAPER_DIR/.placeholder"
      (cd "$OL_DIR" && tar cf - --exclude=./.git .) | (cd "$PAPER_DIR" && tar xf -)
    else
      mirror "$OL_DIR" "$PAPER_DIR"
    fi
    git add -A -f "$PAPER_DIR"
    git commit -q -m "Import paper from Overleaf" || true
    git push -q origin "HEAD:$BRANCH"
    overleaf_commit_push                 # files only GitHub had (figures) go to Overleaf too
  else
    if [[ -f "$PAPER_DIR/.placeholder" ]]; then
      git rm -q "$PAPER_DIR/.placeholder"; git commit -q -m "Starter paper goes to Overleaf"
      git push -q origin "HEAD:$BRANCH"
    fi
    log "the previous Overleaf version stays available in Overleaf: History (top right of the editor)"
    overleaf_commit_push
  fi
  save_state; exit 0
fi

# ------------------------------------------------------------------ regular sync
OL_BASE=$(sed -n 's/^overleaf=//p' "$STATE")
GH_BASE=$(sed -n 's/^github=//p' "$STATE")
changed=0

if [[ $OL_HEAD != "$OL_BASE" ]]; then
  log "Overleaf has new edits"
  git checkout -q -B overleaf-incoming "$GH_BASE"
  mirror "$OL_DIR" "$PAPER_DIR"
  git add -A -f "$PAPER_DIR"
  # Same parent, tree, message and dates every run -> the same commit id. So once a person has
  # merged the overleaf-conflict branch, the next run sees this commit as already merged.
  OL_DATE=$(git -C "$OL_DIR" log -1 --format=%cI)
  GIT_AUTHOR_DATE=$OL_DATE GIT_COMMITTER_DATE=$OL_DATE \
    git commit -q -m "Edits made in Overleaf (Overleaf ${OL_HEAD:0:7})" || true
  git checkout -q "$BRANCH"
  if ! git merge -q --no-edit -m "Merge edits from Overleaf" overleaf-incoming; then
    git merge --abort
    git push -q -f origin overleaf-incoming:overleaf-conflict
    echo "::error::The same lines were edited in Overleaf and on GitHub. The Overleaf version is on the" \
         "branch 'overleaf-conflict': merge it into main, push, and this workflow will finish the sync."
    exit 1
  fi
  git push -q origin --delete overleaf-conflict 2>/dev/null || true   # resolved, if it existed
  changed=1
fi

if ! same_content; then overleaf_commit_push; fi
[[ $changed == 1 ]] && git push -q origin "HEAD:$BRANCH" && log "pushed Overleaf edits to $BRANCH"

if [[ $(git -C "$OL_DIR" rev-parse HEAD) != "$OL_BASE" || $changed == 1 ]] || ! git diff --quiet "$GH_BASE" HEAD -- "$PAPER_DIR"; then
  save_state
else
  log "nothing to sync"
fi
