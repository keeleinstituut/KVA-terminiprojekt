# %%
import json
import os
import sys
from typing import List

import spacy
from document_structure import (Chunk, ContentTextData, Document, FootnoteData,
                             TableData, TermData)
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, PointStruct,
                                       VectorParams)
from sentence_transformers import SentenceTransformer
import argparse
from tqdm import tqdm

# %%
class SpacySenter:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_trf", enable=[
                              'transformer', 'parser'])

    def get_sentences(self, text: str = '') -> List[dict]:
        sentence_data = list()
        for sent in self.nlp(text).sents:
            sentence_data.append({
                'text': sent.text,
                'start_char': sent.start_char,
                'end_char': sent.end_char,
                'length_token': len(sent)
            })

        return sentence_data


class WhitespaceTokenizer:

    def get_tokens(self, text: str = '') -> List[dict]:
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
    

def section_chunks_to_points(document_metadata: dict, section_chunks: List[Chunk], last_idx: int, model):
    section_points = list()

    for i, chunk in enumerate(section_chunks, 1):    
        try:
            chunk_text = 'passage:' + chunk.get_text()
        except TypeError:
            print(chunk)
            print(chunk_text)
            raise TypeError
        payload = document_metadata.copy()
        payload.update(chunk.get_data())
        section_points.append(
                PointStruct(id = last_idx + i,
                            vector=list(model.encode(chunk_text, normalize_embeddings=True).astype(float)),
                            payload=payload)
                            )
    return section_points

if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser(description="Script for adding KVA JSON data to Qdrant vector database-")

    parser.add_argument("arg1", type=str, help="Configuration file")
    parser.add_argument("arg2", type=str, help="Input JSON path")

    args = parser.parse_args()
    try:
        if not args.arg1 or not args.arg2:
            raise ValueError("Error: Missing an argument")
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        if not os.path.isfile(args.arg1):
            raise ValueError("Error: config is not a valid file")
        elif not os.path.isfile(args.arg2):
            raise ValueError("Error: JSON path is not valid")
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

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

    model = SentenceTransformer(embedding_model)
    client = QdrantClient(client_host, port=client_port)

    existing_collections = [coll.name for coll in client.get_collections().collections]

    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )

    tokenizer = WhitespaceTokenizer()
    senter = SpacySenter()

    fpath = args.arg2

    collection_info = client.get_collection(collection_name)

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
            table_extraction_strategy=document_json['table_extraction_strategy'],
            horizontal_sorting=document_json['horizontal_sorting'],
            footnote_regex=document_json['footnote_regex'],
            footnote_group=document_json['footnote_group'],
            custom_regex=document_json['custom_regex'],
            term_data= TermData(document_json['term_data']),
            footnote_data=FootnoteData(document_json['footnote_data']),
            table_data=TableData(document_json['table_data']),
            content_text_data=ContentTextData(document_json['content_text_data'])
            )

        document_metadata = document.get_metadata()

        # parse content chunks one by one 
        content_chunks = document.content_text_data.to_chunks(sentensizer=senter, tokenizer=tokenizer, 
                                                            max_tokens=max_tokens, 
                                                            n_sentences_in_block=sentence_block_size)
        term_chunks = document.term_data.to_chunks()
        footnote_chunks = document.footnote_data.to_chunks()
        table_chunks = document.table_data.to_chunks(tokenizer=tokenizer, max_tokens=max_tokens)


        # Chunks to PointStruct
        for section_chunks in [content_chunks, term_chunks, footnote_chunks, table_chunks]:
            last_idx = client.get_collection(collection_name).vectors_count
            if not last_idx:
                last_idx = 0

            section_points = section_chunks_to_points(document_metadata, section_chunks, last_idx, model=model)
            
            step = 100
            for i in tqdm(range(0, len(section_points), step)): 
                x = i 
                operation_info = client.upsert(
                    collection_name=collection_name,
                    wait=False,
                    points=section_points[x:x+step])


