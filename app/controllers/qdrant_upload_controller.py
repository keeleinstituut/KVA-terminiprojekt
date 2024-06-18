import json
import logging
import os
import uuid
from datetime import datetime
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from app.models.parsed_document_model import Chunk, Document, TextualContent
from utils.db_connection import Connection
from utils.nlp_helpers import E5Tokenizer, SpacySenter
from app.config import config


# Configure logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

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



def upload_vector_data(dir, filename,
                       config,
                       qdrant_client, pg_connection, 
                       embedding_model, 
                       tokenizer, senter):

    logger.info(f'Loading from config: {config}')
    collection_name = config['dbs']['qdrant']['collection_name']
    max_tokens = config['embeddings']['max_tokens']
    passage_prompt = config['embeddings']['passage_prompt']
    sentence_block_size = config['document_chunking']['sentence_block_size']

    # Loading json data
    logger.info(f'Loading json: {os.path.join(dir, filename)}')
    with open(os.path.join(dir, filename), 'r', encoding='utf-8') as fin:
        document_json = json.loads(fin.read())

    logger.info(f'Loaded collection {collection_name}')
    document = Document(json_filename=filename,
                filename=document_json['filename'],
                publication=document_json['publication'],
                publication_year=document_json['publication_year'],
                title=document_json['title'],
                author=document_json['author'],
                languages=document_json['languages'],
                field_keywords=document_json['field_keywords'],
                is_valid=document_json['is_valid'],
                content=TextualContent(document_json['content']))

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
                                                model=embedding_model, passage_prompt=passage_prompt)

    logger.info(f'Embeddings created, starting upload to {collection_name}.')
    logger.setLevel(logging.DEBUG)
    step = 100
    logger.info(f'{len(section_points)} chunks generated for section.')

    for i in range(0, len(section_points), step):
        x = i
        qdrant_client.upsert(
            collection_name=collection_name,
            wait=False,
            points=section_points[x:x+step])
        
    logger.info(f'{len(section_points)} chunks added to database')

    try:
        pg_connection.execute_sql(
            """UPDATE documents
            SET current_state = 'uploaded'
            WHERE pdf_filename = :fname;""",
            {'fname': document.filename}
        )
        pg_connection.commit()
    except Exception as e:
        logger.error(e)


def upload_to_qdrant():
    logging.info('Starting upload to qdrant')

    intermediate_storage_path = config['intermediate_storage']['json_storage_path']
    finished_storage_path = config['intermediate_storage']['finished_json_storage_path']

    filenames = os.listdir(intermediate_storage_path)
    if len(filenames) == 0:
        return None

    logging.info('Accessing configuration values')
    client_host = config['dbs']['qdrant']['host']
    client_port = config['dbs']['qdrant']['port']

    embedding_model = config['embeddings']['embedding_model']

    pg_host = config['dbs']['postgres']['host']
    pg_port = config['dbs']['postgres']['port']
    pg_user = config['dbs']['postgres']['user']
    pg_password = config['dbs']['postgres']['password']
    pg_collection_name = config['dbs']['postgres']['collection_name']

    con = Connection(host=pg_host, port=pg_port, user=pg_user, password=pg_password, db=pg_collection_name)
    engine = con.establish_connection()
    logger.info(f'Established Postgres connection')

    client = QdrantClient(client_host, port=client_port)
    logger.info(f'Connected to qdrant: {client_host}:{client_port}')

    model = SentenceTransformer(embedding_model)
    logger.info(f'Model {embedding_model} loaded.')

    logger.info('Initializing tokenizer and sentence parser.')
    try:
        tokenizer = E5Tokenizer()
        senter = SpacySenter(model='en_core_web_sm')
    except Exception as e:
        logger.error(e)
    logger.info('Tokenizer and parser initialized.')

    for filename in filenames:
        logger.info(f'Starting {filename} upload.')
        upload_vector_data(dir=intermediate_storage_path,
                           config=config,
                           filename=filename,
                           qdrant_client=client,
                           pg_connection=con,
                           embedding_model=model,
                           tokenizer=tokenizer,
                           senter=senter)
        os.rename(os.path.join(intermediate_storage_path, filename), os.path.join(finished_storage_path, filename))
    con.close()
    
