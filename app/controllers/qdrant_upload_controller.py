import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer

from app.models.parsed_document_model import Chunk, Document, TextualContent
from utils.db_connection import Connection
from utils.nlp_helpers import E5Tokenizer
from app.config import config


# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


def encode_sparse(text: str, sparse_model) -> dict:
    """
    Encodes text into a sparse vector using the BM25/SPLADE model.
    
    Args:
        text (str): The text to encode.
        sparse_model: The sparse embedding model.
        
    Returns:
        dict: A dictionary with 'indices' and 'values' for sparse vector.
    """
    embeddings = list(sparse_model.embed([text]))[0]
    return {
        "indices": embeddings.indices.tolist(),
        "values": embeddings.values.tolist()
    }


def section_chunks_to_points(
    document_metadata: dict,
    section_chunks: List[Chunk],
    model,
    passage_prompt: str = "",
    sparse_model=None,
):
    """
    Transforms chunks of document sections into point structures suitable for indexing in a Qdrant vector database.

    This function iterates over a list of document section chunks, encodes them into vectors using a provided model,
    and packages these vectors along with metadata into point structures. Each point structure includes a unique identifier,
    vector representation of the chunk text, and additional metadata such as creation and modification dates.

    Args:
        - collection_name (str): The name of the collection within the Qdrant database where the points will be stored.
        - client (QdrantClient): An instance of the QdrantClient used to interact with the Qdrant database.
        - document_metadata (dict): A dictionary containing metadata that applies to the entire document.
        - section_chunks (List[Chunk]): A list of Chunk objects representing segments of the document.
        - model: A model object capable of encoding text into vector representations.
        - passage_prompt (dict, optional): Additional parameters or prompts to be passed to the model during the encoding process. Defaults to an empty dictionary.
        - sparse_model: Optional sparse embedding model for hybrid search support.

    Returns:
        List[PointStruct]: A list of PointStruct objects, each containing a unique identifier, vector representations (dense and optionally sparse), and the payload of metadata.
    """

    section_points = list()
    previous_chunk_id = None

    for chunk in section_chunks:
        try:
            chunk_text = chunk.get_text()
            if not chunk_text:
                continue
        except TypeError:
            raise TypeError
        payload = document_metadata.copy()
        payload.update(chunk.get_data())
        payload["date_created"] = datetime.date(datetime.today()).isoformat()
        payload["date_modified"] = datetime.date(datetime.today()).isoformat()
        payload["previous_chunk_id"] = previous_chunk_id

        chunk_id = str(uuid.uuid4())
        previous_chunk_id = chunk_id

        # Create dense vector
        dense_vector = list(
            model.encode(
                chunk_text, normalize_embeddings=True, prompt=passage_prompt
            ).astype(float)
        )

        # Create vector dict for named vectors (hybrid search support)
        if sparse_model is not None:
            sparse_vec = encode_sparse(chunk_text, sparse_model)
            vector = {
                "dense": dense_vector,
                "sparse": SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"]
                )
            }
        else:
            # Fallback to single vector for backward compatibility
            vector = dense_vector

        section_points.append(
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload=payload,
            )
        )
    return section_points


def upload_vector_data(
    dir,
    filename: str,
    config: dict,
    qdrant_client: QdrantClient,
    pg_connection: Connection,
    embedding_model: SentenceTransformer,
    tokenizer,
    sparse_model=None,
) -> None:
    """
    Uploads vector data to the Qdrant vector database and updates the document status to 'uploaded' in the PostgreSQL database.

    Args:
        dir (str): The directory path where the JSON file is located.
        filename (str): The name of the JSON file containing the document data.
        config (dict): A dictionary containing configuration settings.
        qdrant_client (QdrantClient): An instance of the QdrantClient used to interact with the Qdrant database.
        pg_connection (Connection): Connection instance used to interact with the PostgreSQL database.
        embedding_model (SentenceTransformer): A pre-trained sentence transformer model for encoding text into vectors.
        tokenizer (E5Tokenizer): An instance of the E5Tokenizer used for tokenizing text.
        sparse_model: Optional sparse embedding model for hybrid search support.
    """

    logger.info(f"Loading from config: {config}")
    collection_name = os.getenv("QDRANT_COLLECTION")
    max_tokens = config["embeddings"]["max_tokens"]
    passage_prompt = config["embeddings"]["passage_prompt"]

    # Loading json data
    logger.info(f"Loading json: {os.path.join(dir, filename)}")
    with open(os.path.join(dir, filename), "r", encoding="utf-8") as fin:
        document_json = json.loads(fin.read())

    logger.info(f"Loaded collection {collection_name}")
    document = Document(
        json_filename=filename,
        filename=document_json["filename"],
        publication=document_json["publication"],
        publication_year=document_json["publication_year"],
        title=document_json["title"],
        author=document_json["author"],
        languages=document_json["languages"],
        field_keywords=document_json["field_keywords"],
        is_valid=document_json["is_valid"],
        content=TextualContent(document_json["content"]),
    )

    document_metadata = document.get_metadata()
    document_metadata["prompt"] = passage_prompt

    # parse content chunks one by one
    logger.info(f"Chunking document data.")
    content_chunks = document.content.to_chunks(
        tokenizer=tokenizer, max_tokens=max_tokens
    )

    # Chunks to PointStruct (with optional sparse embeddings for hybrid search)
    logger.info(f"{len(content_chunks)} chunks generated, Creating embeddings.")
    section_points = section_chunks_to_points(
        document_metadata,
        content_chunks,
        model=embedding_model,
        passage_prompt=passage_prompt,
        sparse_model=sparse_model,
    )

    logger.info(f"Embeddings created, starting upload to {collection_name}.")
    step = 20
    logger.info(f"{len(section_points)} chunks generated for section.")

    for i in range(0, len(section_points), step):
        x = i
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=False,
            points=section_points[x : x + step],
        )

    logger.info(f"{len(section_points)} chunks added to database")

    try:
        logger.info("Starting PG status update.")
        pg_connection.execute_sql(
            """UPDATE documents
            SET current_state = 'uploaded'
            WHERE pdf_filename = :fname;""",
            [{"fname": document.filename}],
        )
        pg_connection.commit()
        logger.info("Finished PG status update.")
    except Exception as e:
        logger.error(e)


def upload_to_qdrant() -> None:
    logging.info("Starting upload to qdrant")

    intermediate_storage_path = config["intermediate_storage"]["json_storage_path"]
    finished_storage_path = config["intermediate_storage"]["finished_json_storage_path"]

    filenames = os.listdir(intermediate_storage_path)
    if len(filenames) == 0:
        return None

    logging.info("Accessing configuration values")
    client_host = os.getenv("QDRANT_HOST")
    client_port = os.getenv("QDRANT_PORT")

    embedding_model = config["embeddings"]["embedding_model"]
    
    # Get hybrid search configuration
    hybrid_config = config.get("hybrid_search", {"enabled": False})

    pg_host = os.getenv("PG_HOST")
    pg_port = os.getenv("PG_PORT")
    pg_user = os.getenv("PG_USER")
    pg_password = os.getenv("PG_PASSWORD")
    pg_collection_name = os.getenv("PG_COLLECTION")

    con = Connection(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        db=pg_collection_name,
    )
    con.establish_connection()

    logger.info(f"Established Postgres connection")

    client = QdrantClient(client_host, port=client_port, prefer_grpc=True)
    logger.info(f"Connected to qdrant: {client_host}:{client_port}")

    model = SentenceTransformer(embedding_model)
    logger.info(f"Model {embedding_model} loaded.")

    # Initialize sparse model for hybrid search if enabled
    sparse_model = None
    if hybrid_config.get("enabled", False):
        try:
            from fastembed import SparseTextEmbedding
            sparse_model_name = hybrid_config.get("sparse_model", "Qdrant/bm25")
            sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
            logger.info(f"Sparse model {sparse_model_name} loaded for hybrid search.")
        except Exception as e:
            logger.warning(f"Failed to load sparse model, falling back to dense-only: {e}")
            sparse_model = None

    logger.info("Initializing tokenizer.")
    try:
        tokenizer = E5Tokenizer()
    except Exception as e:
        logger.error(e)
    logger.info("Tokenizer and parser initialized.")

    for filename in filenames:
        logger.info(f"Starting {filename} upload.")
        upload_vector_data(
            dir=intermediate_storage_path,
            config=config,
            filename=filename,
            qdrant_client=client,
            pg_connection=con,
            embedding_model=model,
            tokenizer=tokenizer,
            sparse_model=sparse_model,
        )
        os.rename(
            os.path.join(intermediate_storage_path, filename),
            os.path.join(finished_storage_path, filename),
        )
    con.close()
