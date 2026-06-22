#!/usr/bin/env bash
# setup_git.sh — merge collectors/ into main repo and set remote

set -e

cd /home/picks

# Get token URL from collectors' remote
PAT_URL=$(git -C /home/picks/collectors remote get-url origin 2>/dev/null || echo "")
if [ -z "$PAT_URL" ]; then
    echo "ERROR: could not get PAT URL from collectors/.git"
    exit 1
fi

echo "=== Setting up main repo remote ==="
git remote remove origin 2>/dev/null || true
git remote add origin "$PAT_URL"
echo "Remote set (token hidden)"

echo ""
echo "=== Removing nested .git from collectors/ ==="
rm -rf /home/picks/collectors/.git
echo "Removed collectors/.git"

echo ""
echo "=== Staging all files ==="
git add -A
echo "Staged files: $(git status --short | wc -l)"
git status --short | head -30

echo ""
echo "=== Creating initial commit ==="
git commit -m "Production snapshot 2026-06-22: WNBA paused, tennis 2nd run added, sync_odds team-name bug fixed, MLB lineup API investigation"

echo ""
echo "=== Pushing to main ==="
git push origin HEAD:main --force
echo "Push complete"
