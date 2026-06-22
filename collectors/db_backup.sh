#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------
# db_backup.sh — weekly picksdb backup to GitHub
# Cron: 0 2 * * 0   (Sunday 2am UTC)
# Location on VPS: /home/picks/collectors/db_backup.sh
# -----------------------------------------------------------------------

REPO_DIR="/home/picks/collectors"
BACKUP_DIR="$REPO_DIR/backups"
DATE=$(date +%Y-%m-%d)
BACKUP_FILE="$BACKUP_DIR/picksdb_${DATE}.sql.gz"

DB_HOST="localhost"
DB_NAME="picksdb"
DB_USER="picksuser"
export PGPASSWORD="password"

cd "$REPO_DIR"

# -- 1. Dump ---------------------------------------------------------------
mkdir -p "$BACKUP_DIR"
echo "[$(date -u +%H:%M:%S)] Running pg_dump..."
pg_dump -h "$DB_HOST" -U "$DB_USER" "$DB_NAME" | gzip > "$BACKUP_FILE"
echo "[$(date -u +%H:%M:%S)] Backup written: $BACKUP_FILE"

# -- 2. Prune: keep only the 8 most recent backups -------------------------
echo "[$(date -u +%H:%M:%S)] Pruning to 8 most recent backups..."
# Sort by filename (date-stamped, so alphabetical = chronological).
# tail -n +9 skips the 8 newest, leaving the older ones to remove.
ls -1 "$BACKUP_DIR"/picksdb_*.sql.gz 2>/dev/null | sort -r | tail -n +9 | while read -r f; do
    echo "  Removing $(basename "$f")"
    git rm --force "$f"
done

# -- 3. Commit and push ----------------------------------------------------
git add "$BACKUP_FILE"

# Only commit if there is something staged
if ! git diff --cached --quiet; then
    git commit -m "db backup $DATE"
    echo "[$(date -u +%H:%M:%S)] Pushing to GitHub..."
    git push origin main
    echo "[$(date -u +%H:%M:%S)] Done."
else
    echo "[$(date -u +%H:%M:%S)] Nothing to commit — backup for $DATE already exists."
fi
