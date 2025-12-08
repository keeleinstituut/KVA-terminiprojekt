"""
Migration script to add URLs from PostgreSQL to existing Qdrant documents.

This script:
1. Reads document URLs from PostgreSQL (title -> URL mapping)
2. Scrolls through all Qdrant points
3. Updates each point's payload with the URL based on the document title

Usage:
    python -m utils.add_urls_to_qdrant
    
    Or from Docker:
    docker compose exec backend python -m utils.add_urls_to_qdrant

Environment variables:
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_COLLECTION
"""

import os
import sys
import logging
from typing import Dict, List

# Add parent directory to path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(_parent_dir, ".env"))

from qdrant_client import QdrantClient
from qdrant_client.http.models import SetPayloadOperation, PointIdsList

from utils.db_connection import Connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_document_urls_from_postgres() -> Dict[str, str]:
    """
    Fetch all document URLs from PostgreSQL.
    
    Returns:
        Dict mapping document title to URL
    """
    logger.info("Connecting to PostgreSQL...")
    
    con = Connection(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
        db=os.getenv("PG_COLLECTION", "postgres"),
    )
    con.establish_connection()
    
    logger.info("Fetching document URLs from PostgreSQL...")
    
    df = con.statement_to_df("SELECT title, url FROM documents WHERE url IS NOT NULL AND url != ''")
    
    url_map = {}
    for _, row in df.iterrows():
        if row['url']:
            url_map[row['title']] = row['url']
    
    con.close()
    
    logger.info(f"Found {len(url_map)} documents with URLs")
    return url_map


def update_qdrant_with_urls(url_map: Dict[str, str], batch_size: int = 50) -> int:
    """
    Update Qdrant points with URLs from the mapping.
    
    Args:
        url_map: Dict mapping document title to URL
        batch_size: Number of points to process per batch
        
    Returns:
        Number of points updated
    """
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    
    logger.info(f"Connecting to Qdrant at {host}:{port}...")
    client = QdrantClient(host=host, port=port)
    
    # Verify collection exists
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        logger.error(f"Collection '{collection_name}' not found!")
        logger.info(f"Available collections: {collections}")
        return 0
    
    logger.info(f"Updating collection '{collection_name}' with URLs...")
    
    # Track stats
    total_updated = 0
    total_skipped = 0
    total_no_match = 0
    
    # Scroll through all points
    offset = None
    batch_num = 0
    
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # We don't need vectors for this update
        )
        
        if not results:
            break
        
        batch_num += 1
        logger.info(f"Processing batch {batch_num} ({len(results)} points)...")
        
        # Group points by URL to update
        updates_by_url = {}
        
        for point in results:
            if not point.payload:
                continue
            
            title = point.payload.get("title", "")
            current_url = point.payload.get("url", "")
            
            # Skip if already has URL
            if current_url:
                total_skipped += 1
                continue
            
            # Look up URL
            if title in url_map:
                url = url_map[title]
                if url not in updates_by_url:
                    updates_by_url[url] = {"title": title, "point_ids": []}
                updates_by_url[url]["point_ids"].append(point.id)
            else:
                total_no_match += 1
        
        # Apply updates
        for url, data in updates_by_url.items():
            point_ids = data["point_ids"]
            if point_ids:
                client.set_payload(
                    collection_name=collection_name,
                    payload={"url": url},
                    points=point_ids,
                )
                total_updated += len(point_ids)
                logger.debug(f"Updated {len(point_ids)} points for '{data['title']}' with URL")
        
        if offset is None:
            break
    
    logger.info(f"Update complete!")
    logger.info(f"  - Updated: {total_updated} points")
    logger.info(f"  - Skipped (already had URL): {total_skipped} points")
    logger.info(f"  - No match (title not in DB): {total_no_match} points")
    
    return total_updated


def main():
    logger.info("=" * 60)
    logger.info("Starting URL migration: PostgreSQL -> Qdrant")
    logger.info("=" * 60)
    
    # Step 1: Get URLs from PostgreSQL
    try:
        url_map = get_document_urls_from_postgres()
    except Exception as e:
        logger.error(f"Failed to fetch URLs from PostgreSQL: {e}")
        return 1
    
    if not url_map:
        logger.warning("No URLs found in PostgreSQL. Nothing to update.")
        return 0
    
    # Log sample of URLs
    logger.info("Sample of document URLs:")
    for i, (title, url) in enumerate(list(url_map.items())[:3]):
        logger.info(f"  {title[:50]}... -> {url[:60]}...")
    
    # Step 2: Update Qdrant
    try:
        updated = update_qdrant_with_urls(url_map)
    except Exception as e:
        logger.error(f"Failed to update Qdrant: {e}")
        return 1
    
    logger.info("=" * 60)
    logger.info("Migration complete!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
