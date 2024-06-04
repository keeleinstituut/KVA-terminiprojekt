import argparse
import json
import logging
import math
import os
import sys
import uuid
from datetime import datetime
from typing import List

import fitz
from app.models.parsed_document_model import Chunk, Document, TextualContent
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from utils.nlp_helpers import E5Tokenizer, SpacySenter
from utils.upload_helpers import reformat_text, normalized_keywords
from utils.db_connection import Connection


def load_data(pdf_file):

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


def section_chunks_to_points(document_metadata: dict,
                             section_chunks: List[Chunk],
                             model,
                             passage_prompt: str = ''):
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

    Returns:
        List[PointStruct]: A list of PointStruct objects, each containing a unique identifier, a vector representation of the text, and the payload of metadata.
    """


    section_points = list()
    previous_chunk_id = None

    for chunk in section_chunks:
        try:
            if not chunk.get_text():
                continue
            chunk_text = 'passage:' + chunk.get_text()
        except TypeError:
            print(chunk)
            print(chunk_text)
            raise TypeError
        payload = document_metadata.copy()
        payload.update(chunk.get_data())
        payload['date_created'] = datetime.date(datetime.today()).isoformat()
        payload['date_modified'] = datetime.date(datetime.today()).isoformat()
        payload['previous_chunk_id'] = previous_chunk_id

        chunk_id = str(uuid.uuid4())
        previous_chunk_id = chunk_id

        section_points.append(
            PointStruct(id=chunk_id,
                        vector=list(model.encode(
                            chunk_text, normalize_embeddings=True, prompt=passage_prompt).astype(float)),
                        payload=payload)
        )
    return section_points


def file_exists_in_collection_old(collection_name, client, filename: str, filename_field: str = 'filename'):
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
    matching_file_entries = client.scroll(
        collection_name=collection_name,  # Replace with your collection name
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=filename_field,
                    match=models.MatchValue(value=filename),
                ),
            ]))[0]

    if len(matching_file_entries) > 0:
        return True
    return False


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
    document_table = con.table_to_dataframe('documents', columns=['pdf_filename'])
    if filename in document_table['pdf_filename']:
        return True
    return False


def upload_to_db(input_pdf, pdf_meta):
    # Parsing arguments
    config_file = 'C:\\Users\\sandra.eiche\\Documents\\git\\KVA-terminiprojekt\\app\\config.json'

    try:
        if not os.path.isfile(config_file):
            raise ValueError(f"Error: {config_file} is not a valid file")
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Load the configuration
    with open(config_file, 'r') as config_file:
        config = json.load(config_file)

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    log_dir =  config['logging']['log_directory_path']

    print(pdf_meta)

    file_handler = logging.FileHandler(os.path.join(
        log_dir, pdf_meta['filename'].rsplit('.', 1)[0] + '.log'))
    
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Accessing configuration values
    client_host = config['dbs']['qdrant']['host']
    client_port = config['dbs']['qdrant']['port']
    collection_name = config['dbs']['qdrant']['collection_name']

    pg_host = config['dbs']['postgres']['host']
    pg_port = config['dbs']['postgres']['port']
    pg_user = config['dbs']['postgres']['user']
    pg_password = config['dbs']['postgres']['password']
    pg_collection_name = config['dbs']['postgres']['collection_name']

    embedding_model = config['embeddings']['embedding_model']
    embedding_size = config['embeddings']['embedding_size']
    max_tokens = config['embeddings']['max_tokens']
    passage_prompt = config['embeddings']['passage_prompt']
    
    sentence_block_size = config['document_chunking']['sentence_block_size']
    
    logger.info("Configuration loaded successfully")

    model = SentenceTransformer(embedding_model)
    logger.info(f'Model {embedding_model} loaded.')

    client = QdrantClient(client_host, port=client_port)
    logger.info(f'Connected to qdrant: {client_host}:{client_port}')

    con = Connection(host=pg_host, port=pg_port, user=pg_user, password=pg_password, db=pg_collection_name)
    engine = con.establish_connection()

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

    # Loading 
    metadata = {
    'json_filename': pdf_meta['filename'].rsplit('.', 1)[0] + '.json',
    'filename': pdf_meta['filename'].strip(),
    'publication': pdf_meta['publication'].strip(),
    'publication_year': pdf_meta['publication_year'],
    'title': pdf_meta['title'].strip(),
    'author': pdf_meta['author'].strip(),
    'languages': pdf_meta['languages'],
    'field_keywords': pdf_meta['keywords'],
    'is_valid': pdf_meta['is_valid']}

    # Checking for duplicate files
    if file_exists_in_collection(con, filename=pdf_meta['filename']):
        logger.error("File %s already exists in collection.",
                        pdf_meta['filename'])
        # TODO: UI error
        raise FileExistsError(
            "File %s already exists in collection.", pdf_meta['filename'])
    
    try:
        # Inital postgres entry
        query = """ INSERT INTO documents (pdf_filename, title, publication, year, author, languages, is_valid, current_state) 
        VALUES (:fname, :title, :publication, :year, :author, :languages, :is_valid, :current_state) """
        cur.execute_sql(
            query,
            [{'fname': metadata['filename'], 
            'title': metadata['title'], 
            'publication': metadata['publication'],
            'year': metadata['publication_year'], 
            'author': metadata['author'], 
            'languages': metadata['languages'],
            'is_valid': metadata['is_valid'], 
            'current_state': 'processing'}])

        # Keywords entry
        kw_query = f""" INSERT INTO keywords (keyword, document_id) VALUES (:kw, :fname) """
        kw_data = [{'kw': kw, 'fname': metadata['filename']} for kw in metadata['field_keywords']]
        con.execute_sql(kw_query, kw_data)
    except Exception as e:
        #pg_connection.rollback()
        logger.error(f"An error occurred: {e}")
    #finally:
        #cur.close()
        #pg_connection.close()

    logger.info('Initializing tokenizer and sentence parser.')
    tokenizer = E5Tokenizer()
    senter = SpacySenter()
    logger.info('Tokenizer and parser initialized.')
    logger.info(f'Loaded collection {collection_name}')

    document = Document(**metadata,
                        content=TextualContent(load_data(input_pdf)))

    document_metadata = document.get_metadata()
    document_metadata['prompt'] = passage_prompt

    # parse content chunks one by one
    logger.info(f'Chunking document data.')
    content_chunks = document.content.to_chunks(sentensizer=senter, tokenizer=tokenizer,
                                                            max_tokens=max_tokens,
                                                            n_sentences_in_block=sentence_block_size)

    # Chunks to PointStruct
    logger.setLevel(logging.CRITICAL)
    logger.info(f'{len(content_chunks)} chunks generated, Creating embeddings.')
    section_points = section_chunks_to_points(document_metadata, content_chunks, 
                                                model=model, passage_prompt=passage_prompt)
    
    logger.info(f'Embeddings created, starting upload to {collection_name}.')
    logger.setLevel(logging.DEBUG)
    step = 100
    logger.info(f'{len(section_points)} chunks generated for section.')
    
    for i in range(0, len(section_points), step):
        x = i
        client.upsert(
            collection_name=collection_name,
            wait=False,
            points=section_points[x:x+step])
        
    logger.info(f'{len(section_points)} chunks added to database')

    con.execute(
        """UPDATE documents
        SET state = 'uploaded'
        WHERE filename = :fname""",
        [{'fname': metadata['filename']}]
    )

