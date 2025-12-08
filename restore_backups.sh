#!/bin/bash

# Restore script for KVA-terminiprojekt databases
# This script restores both PostgreSQL and Qdrant backups

set -e

echo "=== KVA Database Restore Script ==="
echo ""

# Configuration
PG_DUMP="arendajale/dump_kva_siseturvalisus_2025-11-25.sql"
QDRANT_SNAPSHOT="arendajale/kva_siseturvalisus_200_tkn-8324297588151018-2025-11-25-14-03-25.snapshot"
QDRANT_COLLECTION="kva_siseturvalisus_200_tkn"

# Check if backup files exist
if [ ! -f "$PG_DUMP" ]; then
    echo "Error: PostgreSQL dump not found at $PG_DUMP"
    exit 1
fi

if [ ! -f "$QDRANT_SNAPSHOT" ]; then
    echo "Error: Qdrant snapshot not found at $QDRANT_SNAPSHOT"
    exit 1
fi

echo "Starting database containers..."
docker-compose up -d db qdrant

echo "Waiting for PostgreSQL to be ready..."
sleep 5

# Wait for PostgreSQL to accept connections
until docker-compose exec -T db pg_isready -U postgres > /dev/null 2>&1; do
    echo "  Waiting for PostgreSQL..."
    sleep 2
done
echo "PostgreSQL is ready!"

echo ""
echo "=== Restoring PostgreSQL database ==="

# Drop existing tables and types for clean restore
echo "Dropping existing tables..."
docker-compose exec -T db psql -U postgres -d postgres -c "DROP TABLE IF EXISTS keywords, documents CASCADE; DROP TYPE IF EXISTS document_state CASCADE;" 2>/dev/null || true

# Restore the dump
echo "Restoring SQL dump..."
docker-compose exec -T db psql -U postgres -d postgres < "$PG_DUMP"

echo "PostgreSQL restore complete!"

echo ""
echo "=== Restoring Qdrant vector database ==="

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
until curl -s http://localhost:6333/collections > /dev/null 2>&1; do
    echo "  Waiting for Qdrant..."
    sleep 2
done
echo "Qdrant is ready!"

# Restore Qdrant snapshot
echo "Uploading Qdrant snapshot..."
curl -X POST "http://localhost:6333/collections/${QDRANT_COLLECTION}/snapshots/upload" \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@${QDRANT_SNAPSHOT}"

echo ""
echo "Qdrant restore complete!"

echo ""
echo "=== Adding URLs to Qdrant documents ==="
echo ""
echo "Starting backend container temporarily for migration..."
docker-compose up -d backend

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
sleep 15

# Run the URL migration script
echo "Running URL migration..."
docker-compose exec -T backend python -m utils.add_urls_to_qdrant

echo "URL migration complete!"

echo ""
echo "=== All databases restored successfully! ==="
echo ""
echo "You can now start the full application with: docker-compose up -d"
