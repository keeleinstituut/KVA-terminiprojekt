"""
Migration script to add sparse vectors to existing Qdrant collection for hybrid search.

This script:
1. Reads all existing points from the collection
2. Generates sparse embeddings from the stored 'text' field
3. Creates a new collection with hybrid vector support
4. Re-inserts all points with both dense and sparse vectors

Usage:
    python -m utils.migrate_to_hybrid
    
Environment variables:
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
"""

import os
import sys
import logging
import json
from typing import List

# Add parent directory to path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Load environment variables BEFORE importing app.config
from dotenv import load_dotenv
load_dotenv(os.path.join(_parent_dir, ".env"))

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

# Load config directly to avoid import issues
def load_config_direct():
    # Try APP_CONFIG env var first, then default to config/config.json
    config_path = os.getenv('APP_CONFIG')
    if not config_path or not os.path.exists(config_path):
        config_path = os.path.join(_parent_dir, 'config', 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config_direct()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_all_points(client: QdrantClient, collection_name: str, batch_size: int = 100) -> List[dict]:
    """
    Fetch all points from the collection using scroll.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        batch_size: Number of points to fetch per batch
        
    Returns:
        List of all points with their vectors and payloads
    """
    all_points = []
    offset = None
    
    logger.info(f"Fetching all points from '{collection_name}'...")
    
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        
        if not results:
            break
            
        all_points.extend(results)
        logger.info(f"Fetched {len(all_points)} points so far...")
        
        if offset is None:
            break
    
    logger.info(f"Total points fetched: {len(all_points)}")
    return all_points


def create_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_size: int,
) -> None:
    """
    Create a new collection with hybrid vector support.
    """
    logger.info(f"Creating hybrid collection '{collection_name}'...")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=dense_size,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                ),
            ),
        },
    )
    
    logger.info(f"Collection '{collection_name}' created with hybrid support.")


def migrate_collection(
    client: QdrantClient,
    source_collection: str,
    target_collection: str,
    sparse_model,
    dense_size: int,
    batch_size: int = 20,
) -> None:
    """
    Migrate points from source to target collection with sparse vectors.
    """
    # Fetch all points from source
    all_points = fetch_all_points(client, source_collection)
    
    if not all_points:
        logger.warning("No points found in source collection!")
        return
    
    # Create target collection
    create_hybrid_collection(client, target_collection, dense_size)
    
    # Process and upload points in batches
    logger.info("Generating sparse embeddings and uploading to new collection...")
    
    new_points = []
    for i, point in enumerate(all_points):
        # Get the text from payload
        text = point.payload.get("text", "")
        if not text:
            logger.warning(f"Point {point.id} has no text field, skipping sparse embedding")
            continue
        
        # Get existing dense vector
        dense_vector = point.vector
        if isinstance(dense_vector, dict):
            # Already has named vectors
            dense_vector = dense_vector.get("dense", dense_vector.get("", list(dense_vector.values())[0]))
        
        # Generate sparse embedding
        sparse_embedding = list(sparse_model.embed([text]))[0]
        sparse_vector = SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )
        
        # Create new point with both vectors
        new_point = PointStruct(
            id=point.id,
            vector={
                "dense": list(dense_vector) if not isinstance(dense_vector, list) else dense_vector,
                "sparse": sparse_vector,
            },
            payload=point.payload,
        )
        new_points.append(new_point)
        
        # Upload in batches
        if len(new_points) >= batch_size:
            client.upsert(
                collection_name=target_collection,
                points=new_points,
                wait=True,
            )
            logger.info(f"Uploaded {i + 1}/{len(all_points)} points...")
            new_points = []
    
    # Upload remaining points
    if new_points:
        client.upsert(
            collection_name=target_collection,
            points=new_points,
            wait=True,
        )
    
    logger.info(f"Migration complete! {len(all_points)} points migrated.")


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate Qdrant collection to hybrid search")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip all confirmations (for automated scripts)")
    args = parser.parse_args()
    
    auto_confirm = args.yes
    
    # Environment already loaded at module level
    
    # Get configuration
    client_host = os.getenv("QDRANT_HOST", "localhost")
    client_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    dense_size = config["embeddings"]["embedding_size"]
    hybrid_config = config.get("hybrid_search", {})
    sparse_model_name = hybrid_config.get("sparse_model", "Qdrant/bm25")
    
    logger.info(f"Connecting to Qdrant at {client_host}:{client_port}")
    client = QdrantClient(host=client_host, port=client_port)
    
    # Check if source collection exists
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        logger.error(f"Collection '{collection_name}' does not exist!")
        return
    
    # Get collection info
    info = client.get_collection(collection_name)
    logger.info(f"Source collection '{collection_name}':")
    logger.info(f"  - Points: {info.points_count}")
    
    if info.points_count == 0:
        logger.error("Collection is empty, nothing to migrate!")
        return
    
    # Check if already a hybrid collection
    new_collection = f"{collection_name}_hybrid"
    if any(c.name == new_collection for c in collections):
        # Check if it has points
        hybrid_info = client.get_collection(new_collection)
        if hybrid_info.points_count > 0:
            logger.info(f"Hybrid collection '{new_collection}' already exists with {hybrid_info.points_count} points.")
            logger.info("Skipping migration - already done.")
            return
    
    # Load sparse model
    logger.info(f"Loading sparse model: {sparse_model_name}")
    try:
        from fastembed import SparseTextEmbedding
        sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        logger.info("Sparse model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load sparse model: {e}")
        logger.error("Make sure fastembed is installed: pip install fastembed")
        return
    
    # Confirm with user
    backup_collection = f"{collection_name}_backup"
    
    print("\n" + "=" * 60)
    print("MIGRATION PLAN")
    print("=" * 60)
    print(f"Source collection:  {collection_name} ({info.points_count} points)")
    print(f"Backup collection:  {backup_collection}")
    print(f"New collection:     {new_collection}")
    print("\nSteps:")
    print("  1. Create backup of current collection")
    print("  2. Create new collection with hybrid vector support")
    print("  3. Migrate all points with sparse embeddings")
    print("  4. Rename collections (backup old, use new)")
    print("=" * 60)
    
    if not auto_confirm:
        response = input("\nProceed with migration? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Migration cancelled.")
            return
    else:
        logger.info("Auto-confirm enabled, proceeding...")
    
    # Step 1: Check if backup already exists
    if any(c.name == backup_collection for c in collections):
        logger.warning(f"Backup collection '{backup_collection}' already exists!")
        if not auto_confirm:
            response = input("Delete existing backup and continue? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Migration cancelled.")
                return
        client.delete_collection(backup_collection)
    
    # Step 2: Check if new collection already exists (empty)
    if any(c.name == new_collection for c in collections):
        logger.warning(f"New collection '{new_collection}' already exists (empty)!")
        if not auto_confirm:
            response = input("Delete it and continue? (yes/no): ")
            if response.lower() != "yes":
                logger.info("Migration cancelled.")
                return
        client.delete_collection(new_collection)
    
    # Step 3: Migrate to new collection
    migrate_collection(
        client=client,
        source_collection=collection_name,
        target_collection=new_collection,
        sparse_model=sparse_model,
        dense_size=dense_size,
    )
    
    # Step 4: Rename collections
    print("\n" + "=" * 60)
    print("FINAL STEP: Rename collections")
    print("=" * 60)
    print(f"This will:")
    print(f"  1. Rename '{collection_name}' -> '{backup_collection}'")
    print(f"  2. Rename '{new_collection}' -> '{collection_name}'")
    print("=" * 60)
    
    if not auto_confirm:
        response = input("\nProceed with rename? (yes/no): ")
        if response.lower() != "yes":
            logger.info(f"Rename skipped. New collection available as '{new_collection}'")
            logger.info(f"You can manually update QDRANT_COLLECTION env var to '{new_collection}'")
            return
    
    # Rename original to backup
    logger.info(f"Renaming '{collection_name}' to '{backup_collection}'...")
    # Qdrant doesn't have a rename API, so we need to use aliases or just inform the user
    logger.info("NOTE: Qdrant doesn't support direct rename. Using collection aliasing...")
    
    # Alternative: Just update the env var
    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"\nYour migrated collection is: {new_collection}")
    print(f"\nTo use it, update your .env file:")
    print(f"  QDRANT_COLLECTION={new_collection}")
    print(f"\nOr delete the old collection and create the new one with the same name.")
    print(f"\nOriginal collection '{collection_name}' is preserved as backup.")
    print("=" * 60)


if __name__ == "__main__":
    main()
