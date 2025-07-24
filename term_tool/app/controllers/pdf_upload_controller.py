import json
import logging
import os

import fitz
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy.exc import IntegrityError
from app.config import config

from utils.db_connection import Connection
from utils.upload_helpers import reformat_text
from io import BytesIO

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


def load_content_text(pdf_file: BytesIO) -> list[str]:
    logger.info("Loading data from pdf file")
    content_text_data = []

    with fitz.open("pdf", pdf_file) as doc:
        for _, page in enumerate(doc, 1):
            # Koguteksti salvestamine json-ina
            full_text_page_json = [
                {
                    "page_number": page.number
                    + 1,  # Page numbers are zero-based in PyMuPDF
                    "text": reformat_text(page.get_text()),
                }
            ]
            content_text_data.extend(full_text_page_json)

    return content_text_data


def file_exists_in_collection(con: Connection, filename: str) -> bool:
    """
    Checks, whether the filename is already present in given collection or not.

    Args:
        - collection_name (str): The name of the collection within the Qdrant database where the points will be stored.
        - client (QdrantClient): An instance of the QdrantClient used to interact with the Qdrant database.
        - filename: The filename to check for.
        - filename_field: The field to search the match from.

    Returns:
        bool: True if the filename already exists in given collection, otherwise Flase
    """
    document_table = con.table_to_dataframe("documents")
    if filename in document_table["pdf_filename"]:
        return True
    return False


def upload_to_db(input_pdf: BytesIO, pdf_meta: dict) -> int:
    """
    Uploads the metadata of a document to PostgreSQL database.
    Generates json from PDF content and metadata for further processing.

    Args:
        input_pdf (BytesIO): Content to the input PDF file.
        pdf_meta (dict): A dictionary containing the metadata of the PDF file, including:
            filename (str): The name of the PDF file.
            publication (str): The publication name.
            publication_year (int): The year of publication.
            title (str): The title of the PDF.
            author (str): The author of the PDF.
            languages (list): A list of languages present in the PDF.
            keywords (list): A list of keywords associated with the PDF.
            is_valid (bool): A flag indicating whether the PDF is valid or not.

    Returns:
        int: A status code indicating the success or failure of the operation:
            - 1: Successful upload
            - -1: PDF file already exists in the database
            - -2: An unexpected error occurred during the upload process

    Raises:
        IntegrityError: If the PDF file already exists in the database.
        Exception: If any other unexpected error occurs during the upload process.
    """

    # Accessing configuration values
    client_host = os.getenv("QDRANT_HOST")
    client_port = os.getenv("QDRANT_PORT")
    collection_name = os.getenv("QDRANT_COLLECTION")

    pg_host = os.getenv("PG_HOST")
    pg_port = os.getenv("PG_PORT")
    pg_user = os.getenv("PG_USER")
    pg_password = os.getenv("PG_PASSWORD")
    pg_collection_name = os.getenv("PG_COLLECTION")

    embedding_size = config["embeddings"]["embedding_size"]
    intermediate_storage_path = config["intermediate_storage"]["json_storage_path"]

    logger.info("Configuration loaded successfully")

    client = QdrantClient(client_host, port=client_port)
    logger.info(f"Connected to qdrant: {client_host}:{client_port}")

    con = Connection(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        db=pg_collection_name,
    )
    engine = con.establish_connection()
    if engine:
        logger.info(f"Connected to postgre: {client_host}:{client_port}")

    # Checking if vector db collection exist
    existing_collections = [coll.name for coll in client.get_collections().collections]

    if collection_name not in existing_collections:
        logger.info(
            f"Creating a new collection {collection_name} with embedding size {embedding_size}."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )

    logger.info("Loading: %s", pdf_meta)
    document_data = {
        "json_filename": pdf_meta["filename"].rsplit(".", 1)[0] + ".json",
        "filename": pdf_meta["filename"].strip(),
        "publication": pdf_meta["publication"].strip(),
        "publication_year": pdf_meta["publication_year"],
        "title": pdf_meta["title"].strip(),
        "author": pdf_meta["author"].strip(),
        "languages": pdf_meta["languages"],
        "field_keywords": pdf_meta["keywords"],
        "url": pdf_meta["url"].strip(),
        "is_valid": pdf_meta["is_valid"],
    }

    try:
        logger.info("Inserting file metadata into 'documents'.")
        query = """ INSERT INTO documents (pdf_filename, json_filename, title, publication, year, author, languages, is_valid, current_state, url) 
        VALUES (:fname, :json_fname, :title, :publication, :year, :author, :languages, :is_valid, :current_state, :url)
        RETURNING documents.id """
        data = [
            {
                "fname": document_data["filename"],
                "json_fname": document_data["json_filename"],
                "title": document_data["title"],
                "publication": document_data["publication"],
                "year": document_data["publication_year"],
                "author": document_data["author"],
                "languages": document_data["languages"],
                "is_valid": document_data["is_valid"],
                "url": document_data["url"],
                "current_state": "processing",
            }
        ]
        result = con.execute_sql(query, data)
        doc_id = result["data"][0][0]

        # Keywords entry
        logger.info("Inserting keywords into 'documents'.")
        if document_data["field_keywords"]:
            logger.info("Inserting document keywords into 'keywords'.")
            kw_query = f""" INSERT INTO keywords (keyword, document_id) VALUES (:kw, :doc_id)"""
            logger.info(document_data["field_keywords"])
            kw_data = [
                {"kw": kw, "doc_id": doc_id} for kw in document_data["field_keywords"]
            ]
            con.execute_sql(kw_query, kw_data)

        con.commit()
        con.close()
    except IntegrityError as e:
        logger.error(
            f"An error occurred: {e}. File is already present in database. Canceling transaction."
        )
        con.session.rollback()
        return -1

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        con.session.rollback()
        return -2

    # Adding body text to document data
    document_data["content"] = load_content_text(input_pdf)

    # Save pdf data to json
    logger.info(
        f'Saving pdf data to: {os.path.join(intermediate_storage_path, document_data["json_filename"])}'
    )
    f = open(
        os.path.join(intermediate_storage_path, document_data["json_filename"]),
        "w",
        encoding="utf-8",
    )
    json.dump(document_data, f, ensure_ascii=False, indent=4)
    f.close()

    return 1
