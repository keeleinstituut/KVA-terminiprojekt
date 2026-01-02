"""
Utility script for setting up Qdrant collection with hybrid search support.

This script creates a new collection or recreates an existing one with:
- Dense vectors: For semantic similarity search (E5 embeddings)
- Sparse vectors: For keyword/BM25-style search

IMPORTANT: Running this script will DELETE existing data in the collection!
Make sure to backup your data before running this script.

Usage:
    python -m utils.qdrant_collection_setup
    
    Or with environment variables:
    QDRANT_HOST=localhost QDRANT_PORT=6333 QDRANT_COLLECTION=my_collection python -m utils.qdrant_collection_setup
"""

import os
import sys
import logging

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

# Add parent directory to path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from app.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_hybrid_collection(
    client: QdrantClient,
    collection_name: str,
    dense_size: int = 1024,
    recreate: bool = False,
) -> bool:
    """
    Set up a Qdrant collection with hybrid search support (dense + sparse vectors).
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to create
        dense_size: Dimension of dense vectors (default: 1024 for E5-large)
        recreate: If True, delete existing collection and create new one
        
    Returns:
        bool: True if collection was created/already exists with correct config
    """
    
    # Check if collection exists
    collections = client.get_collections().collections
    collection_exists = any(c.name == collection_name for c in collections)
    
    if collection_exists and not recreate:
        logger.info(f"Collection '{collection_name}' already exists.")
        logger.info("Use recreate=True to delete and recreate the collection.")
        logger.info("NOTE: Existing data may not have sparse vectors. Re-upload documents for hybrid search.")
        return True
    
    if collection_exists and recreate:
        logger.warning(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted.")
    
    # Create collection with named vectors for hybrid search
    logger.info(f"Creating collection '{collection_name}' with hybrid search support...")
    
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
    
    logger.info(f"Collection '{collection_name}' created successfully with:")
    logger.info(f"  - Dense vectors: size={dense_size}, distance=COSINE")
    logger.info(f"  - Sparse vectors: BM25/keyword search enabled")
    
    return True


def check_collection_config(client: QdrantClient, collection_name: str) -> dict:
    """
    Check the current configuration of a collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to check
        
    Returns:
        dict: Collection configuration info
    """
    try:
        info = client.get_collection(collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "config": info.config,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def main():
    """Main function to set up the Qdrant collection."""
    load_dotenv(".env")
    
    # Get configuration
    client_host = os.getenv("QDRANT_HOST", "localhost")
    client_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    dense_size = config["embeddings"]["embedding_size"]
    
    logger.info(f"Connecting to Qdrant at {client_host}:{client_port}")
    client = QdrantClient(host=client_host, port=client_port)
    
    # Check current collection status
    current_config = check_collection_config(client, collection_name)
    
    if current_config["exists"]:
        logger.info(f"Current collection '{collection_name}' status:")
        logger.info(f"  - Points: {current_config['points_count']}")
        logger.info(f"  - Vectors: {current_config['vectors_count']}")
        
        # Ask user for confirmation before recreating
        if current_config["points_count"] > 0:
            print("\n⚠️  WARNING: Collection has existing data!")
            print("Recreating the collection will DELETE all existing data.")
            response = input("Do you want to recreate the collection? (yes/no): ")
            
            if response.lower() != "yes":
                logger.info("Operation cancelled. Collection unchanged.")
                return
            
            setup_hybrid_collection(
                client, collection_name, dense_size=dense_size, recreate=True
            )
        else:
            logger.info("Collection is empty. Recreating with hybrid search support.")
            setup_hybrid_collection(
                client, collection_name, dense_size=dense_size, recreate=True
            )
    else:
        logger.info(f"Collection '{collection_name}' does not exist. Creating new collection.")
        setup_hybrid_collection(
            client, collection_name, dense_size=dense_size, recreate=False
        )
    
    logger.info("\n✅ Collection setup complete!")
    logger.info("Next steps:")
    logger.info("  1. Re-upload your documents to generate both dense and sparse embeddings")
    logger.info("  2. Ensure 'hybrid_search.enabled' is set to true in config/config.json")


if __name__ == "__main__":
    main()
