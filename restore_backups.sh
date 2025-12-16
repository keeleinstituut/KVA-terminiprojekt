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
echo "Applying SQL migrations..."
for migration in db/migrations/*.sql; do
    echo "  Applying ${migration}..."
    docker-compose exec -T db psql -U postgres -d postgres < "$migration"
done

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
echo "=== Starting backend for migrations ==="
echo ""
echo "Starting backend container..."
docker-compose up -d backend

# Wait for backend to be ready
echo "Waiting for backend to initialize (loading ML models)..."
sleep 30

# Check if backend is healthy
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "  Waiting for backend..."
    sleep 5
done
echo "Backend is ready!"

echo ""
echo "=== Migrating to hybrid search ==="
echo ""
echo "This creates a new collection with dense+sparse vectors..."
# Run migration with the restored collection name (not the .env hybrid name)
docker-compose exec -T -e QDRANT_COLLECTION="${QDRANT_COLLECTION}" backend python -m utils.migrate_to_hybrid --yes

echo ""
echo "=== Adding URLs to Qdrant documents ==="
echo ""
echo "Running URL migration on hybrid collection..."
# Run URL migration on the new hybrid collection
docker-compose exec -T -e QDRANT_COLLECTION="${QDRANT_COLLECTION}_hybrid" backend python -m utils.add_urls_to_qdrant

echo "URL migration complete!"

echo ""
echo "=== All databases restored successfully! ==="
echo ""
echo "Collections available:"
curl -s http://localhost:6333/collections | python3 -c "import sys,json; [print(f'  - {c[\"name\"]}') for c in json.load(sys.stdin)['result']['collections']]"
echo ""
echo "Make sure your .env file has:"
echo "  QDRANT_COLLECTION=${QDRANT_COLLECTION}_hybrid"
echo ""
echo "You can now start the full application with: docker-compose up -d"
