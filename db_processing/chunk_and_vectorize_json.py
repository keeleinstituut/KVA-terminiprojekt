# %%
import argparse
import json
import logging
import math
import os
import sys
import uuid
from datetime import datetime
from typing import List

import spacy
from document_structure import (Chunk, ContentTextData, Document, FootnoteData,
                                TableData, TermData, divide_chunks)
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer


# %%
class SpacySenter:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_trf", enable=[
                              'transformer', 'parser'])

    def get_sentences(self, text: str = '') -> List[dict]:
        sentence_data = list()

        # Spacy jaoks liigade pikkade tekstide jaotamine
        if len(text) >= 1000000:
            new_block_size = round(len(text) / math.ceil(
                len(text) / 1000000))  # for more equal chunks
            text_blocks = divide_chunks(text, new_block_size)
        else:
            text_blocks = [text]

        # lausestamine
        for block in text_blocks:
            sents = self.nlp(block).sents
            for sent in sents:
                sentence_data.append({
                    'text': sent.text,
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'length_token': len(sent)
                })

        return sentence_data


class WhitespaceTokenizer:

    def get_tokens(self, text: str = '') -> List[dict]:
        """
        Tokenizes based on whitespace and returns list of dictionaries with text, start_char and end_char for each token.
        """
        token_data = list()

        tokens = text.split(' ')

        char_counter = 0

        for token in tokens:
            token_data.append({
                'text': token,
                'start_char': char_counter,
                'end_char': char_counter + len(token),
            })
            char_counter += len(token) + 1
        return token_data


class E5Tokenizer:


    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            'intfloat/multilingual-e5-large')

    def get_tokens(self, text: str = '') -> List[dict]:
        """
        Tokenizes using 'intfloat/multilingual-e5-large' and returns list of dictionaries with text, start_char and end_char for each token.
        """

        token_data = list()

        encoded_input = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False, verbose=False)

        # The 'offset_mapping' contains the start and end positions of each token in the original text
        offset_mapping = encoded_input['offset_mapping']

        for token_index, (start_pos, end_pos) in enumerate(offset_mapping):
            token_data.append({
                'text': text[start_pos:end_pos],
                'start_char': start_pos,
                'end_char': end_pos,
            })

        return token_data


def section_chunks_to_points(collection_name: str,
                             client: QdrantClient,
                             document_metadata: dict,
                             section_chunks: List[Chunk],
                             model,
                             passage_prompt: dict = {}):
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

    for i, chunk in enumerate(section_chunks, 1):
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

        chunk_id = str(uuid.uuid4())

        while id_exists(collection_name, client, chunk_id):
            chunk_id = str(uuid.uuid4())

        section_points.append(
            PointStruct(id=chunk_id,
                        vector=list(model.encode(
                            chunk_text, normalize_embeddings=True, prompt=passage_prompt).astype(float)),
                        payload=payload)
        )
    return section_points


def id_exists(collection_name, client, idx):
    """
    Checks, whether an ID exists in given collection or not.

    Args:
        - collection_name (str): The name of the collection within the Qdrant database where the points will be stored.
        - client (QdrantClient): An instance of the QdrantClient used to interact with the Qdrant database.
        - id(int or str): The identifier to check for	

    Returns:
        bool: True if the index already exists in given collection, otherwise Flase
    """
    matching_id_entries = client.scroll(
        collection_name=collection_name,  # Replace with your collection name
        scroll_filter=models.Filter(
            must=[
                models.HasIdCondition(has_id=[idx])  # Filter by ID
            ]))[0]

    if len(matching_id_entries) > 0:
        return True
    return False


def file_exists(collection_name, client, filename: str, filename_field: str = 'filename'):
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


if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser(
        description="Script for adding KVA JSON data to Qdrant vector database-")

    parser.add_argument("arg1", type=str, help="Configuration file")
    parser.add_argument("arg2", type=str, help="Input JSON path")
    parser.add_argument("arg3", type=str, help="Log directory")

    args = parser.parse_args()

    try:
        if not args.arg1 or not args.arg2:
            raise ValueError("Error: Missing an argument")
    except ValueError as e:
        print(e)
        sys.exit(1)

    try:
        if not os.path.isfile(args.arg1):
            print(args.arg1)
            raise ValueError("Error: config is not a valid file")
        elif not os.path.isfile(args.arg2):
            raise ValueError(f"Error: JSON path '{args.arg2}' is not valid")
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(
        args.arg3, args.arg2.split('/')[-1].split('.')[0] + '.log'))
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Load the configuration
    with open(args.arg1, 'r') as config_file:
        config = json.load(config_file)

    # Accessing configuration values
    embedding_model = config['embedding_model']
    client_host = config['client']['host']
    client_port = config['client']['port']
    collection_name = config['collection_name']
    embedding_size = config['embedding_size']
    max_tokens = config['max_tokens']
    sentence_block_size = config['sentence_block_size']
    passage_prompt = config['passage_prompt']
    logger.info("Configuration loaded successfully")

    model = SentenceTransformer(embedding_model)
    logger.info(f'Model {embedding_model} loaded.')

    client = QdrantClient(client_host, port=client_port)
    logger.info(f'Connected to qdrant: {client_host}:{client_port}')

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

    logger.info('Initializing tokenizer and sentence parser.')
    tokenizer = E5Tokenizer()
    senter = SpacySenter()
    logger.info('Tokenizer and parser initialized.')

    fpath = args.arg2

    collection_info = client.get_collection(collection_name)
    logger.info(f'Loaded collection {collection_name}')

    with open(fpath, 'r') as fin:

        document_json = json.loads(fin.read())

        document = Document(
            json_filename=fpath.split('\\')[-1],
            filename=document_json['filename'],
            publication=document_json['publication'],
            publication_year=document_json['publication_year'],
            title=document_json['title'],
            author=document_json['author'],
            languages=document_json['languages'],
            field_keywords=document_json['field_keywords'],
            header_height=document_json['header_height'],
            footer_height=document_json['footer_height'],
            # document_json['table_extraction_strategy'],
            table_extraction_strategy='None',
            horizontal_sorting=True,  # document_json['horizontal_sorting'],
            footnote_regex='',  # document_json['footnote_regex'],
            footnote_group=0,  # document_json['footnote_group'],
            custom_regex=document_json['custom_regex'],
            term_data=TermData(document_json['term_data']),
            footnote_data=FootnoteData(document_json['footnote_data']),
            table_data=TableData(document_json['table_data']),
            content_text_data=ContentTextData(
                document_json['content_text_data'])
        )

        document_metadata = document.get_metadata()
        document_metadata['prompt'] = passage_prompt

        # Check if a Document with the same filename already exists
        if file_exists(client=client, collection_name=collection_name, filename=document.filename):
            logger.error("File %s already exists in collection.",
                         document.filename)
            raise FileExistsError(
                "File %s already exists in collection.", document.filename)

        # parse content chunks one by one
        content_chunks = document.content_text_data.to_chunks(sentensizer=senter, tokenizer=tokenizer,
                                                              max_tokens=max_tokens,
                                                              n_sentences_in_block=sentence_block_size)
        term_chunks = document.term_data.to_chunks()
        footnote_chunks = document.footnote_data.to_chunks()
        table_chunks = document.table_data.to_chunks(
            tokenizer=tokenizer, max_tokens=max_tokens)

        # Chunks to PointStruct
        for i, section_chunks in tqdm(enumerate([content_chunks, term_chunks, footnote_chunks, table_chunks])):
            logger.info(f'Logging section {i}')

            section_points = section_chunks_to_points(collection_name, client, document_metadata,
                                                      section_chunks, passage_prompt=passage_prompt, model=model)

            step = 100
            for i in range(0, len(section_points), step):
                x = i
                operation_info = client.upsert(
                    collection_name=collection_name,
                    wait=False,
                    points=section_points[x:x+step])
            logger.info(f'{len(section_points)} chunks added to database')
