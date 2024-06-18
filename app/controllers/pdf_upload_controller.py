import json
import logging
import os
import sys

import fitz
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from sqlalchemy.exc import IntegrityError
from app.config import config

from utils.db_connection import Connection
from utils.upload_helpers import reformat_text

# Configure logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

def load_content_text(pdf_file):
    logger.info('Loading data from pdf file')
    content_text_data = []

    with fitz.open('pdf', pdf_file) as doc:
        for _, page in enumerate(doc, 1):
            # Koguteksti salvestamine json-ina
            full_text_page_json = [{
                "page_number": page.number + 1,  # Page numbers are zero-based in PyMuPDF
                "text": reformat_text(page.get_text())
                }]
            content_text_data.extend(full_text_page_json)

    return content_text_data


def file_exists_in_collection(con, filename: str):
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
    document_table = con.table_to_dataframe('documents')
    if filename in document_table['pdf_filename']:
        return True
    return False


def upload_to_db(input_pdf, pdf_meta):

    # Accessing configuration values
    client_host = config['dbs']['qdrant']['host']
    client_port = config['dbs']['qdrant']['port']
    collection_name = config['dbs']['qdrant']['collection_name']

    pg_host = config['dbs']['postgres']['host']
    pg_port = config['dbs']['postgres']['port']
    pg_user = config['dbs']['postgres']['user']
    pg_password = config['dbs']['postgres']['password']
    pg_collection_name = config['dbs']['postgres']['collection_name']

    embedding_size = config['embeddings']['embedding_size']
    intermediate_storage_path = config['intermediate_storage']['json_storage_path']
    
    
    logger.info("Configuration loaded successfully")

    client = QdrantClient(client_host, port=client_port)
    logger.info(f'Connected to qdrant: {client_host}:{client_port}')

    con = Connection(host=pg_host, port=pg_port, user=pg_user, password=pg_password, db=pg_collection_name)
    engine = con.establish_connection()
    if engine:
        logger.info(f'Connected to postgre: {client_host}:{client_port}')

    # Checking if vector db collection exist
    existing_collections = [
        coll.name for coll in client.get_collections().collections]

    if collection_name not in existing_collections:
        logger.info(
            f'Creating a new collection {collection_name} with embedding size {embedding_size}.')
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size, distance=Distance.COSINE),
        )

    logger.info('Loading: %s', pdf_meta)
    document_data = {
    'json_filename': pdf_meta['filename'].rsplit('.', 1)[0] + '.json',
    'filename': pdf_meta['filename'].strip(),
    'publication': pdf_meta['publication'].strip(),
    'publication_year': pdf_meta['publication_year'],
    'title': pdf_meta['title'].strip(),
    'author': pdf_meta['author'].strip(),
    'languages': pdf_meta['languages'],
    'field_keywords': pdf_meta['keywords'],
    'is_valid': pdf_meta['is_valid']
    }
    
    try:
        logger.info("Inserting file metadata into 'documents'.")
        # Inital postgres entry
        query = """ INSERT INTO documents (pdf_filename, json_filename, title, publication, year, author, languages, is_valid, current_state) 
        VALUES (:fname, :json_fname, :title, :publication, :year, :author, :languages, :is_valid, :current_state)
        RETURNING documents.id """
        data = [{'fname': document_data['filename'], 
            'json_fname': document_data['json_filename'], 
            'title': document_data['title'], 
            'publication': document_data['publication'],
            'year': document_data['publication_year'], 
            'author': document_data['author'], 
            'languages': document_data['languages'],
            'is_valid': document_data['is_valid'], 
            'current_state': 'processing'}]   
        result = con.execute_sql(query, data)
        doc_id = result[0][0]
        
        # Keywords entry
        logger.info("Inserting keywords into 'documents'.")
        if document_data['field_keywords']:
            logger.info("Inserting document keywords into 'keywords'.")
            kw_query = f""" INSERT INTO keywords (keyword, document_id) VALUES (:kw, :doc_id)"""
            logger.info(document_data['field_keywords'])
            kw_data = [{'kw': kw, 'doc_id': doc_id} for kw in document_data['field_keywords']]
            con.execute_sql(kw_query, kw_data)

        con.commit()
        con.close()
    except IntegrityError as e:
        logger.error(f"An error occurred: {e}. File is already present in database. Canceling transaction.")
        return -1
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return -2

    # Adding body text to document data
    document_data['content'] = load_content_text(input_pdf)
    
    # Save pdf data to json
    logger.info(f'Saving pdf data to: {os.path.join(intermediate_storage_path, document_data["json_filename"])}' )
    f = open(os.path.join(intermediate_storage_path, document_data['json_filename']), 'w', encoding='utf-8')
    json.dump(document_data, f, ensure_ascii=False, indent=4)
    f.close()

    return 1