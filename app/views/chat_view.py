import panel as pn
from panel.chat import ChatInterface
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import (FieldCondition, Filter, MatchAny,
                                       MatchValue)
from collections import defaultdict
import asyncio
from app.config import config



class QdrantChat():

    target_files = []

    response_limit = 5

    chunk_validity_filter = FieldCondition(
                key="validated",
                match=MatchValue(
                    value=True,
                ),)
    
    file_filter = FieldCondition(
        key="filename",
        match=MatchAny(any=target_files),)


    apply_chunk_validity_filter = True
    apply_file_filter = False
    
    def __init__(self, model_name, collection_name) -> None:
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name

    def connect(self,host, port):
        self.host = host
        self.port = port
        self.client = QdrantClient(host, port=port)

    
    async def chat_callback(self, contents, user, instance):
        await asyncio.sleep(1.8)
        db_filter = self.assemble_filter()
        results = get_similarities(self.client, self.model, contents, 
                                db_filter, collection_name=self.collection_name,
                                response_limit=self.response_limit)
        
        response = ''

        for result in zip(results['response_text'], results['filename'], results['page_no']):
            response += f'File: {result[1]}\nPage: {result[2]}\n\n{result[0]}\n{"*"*15}\n'

        return response
    
    def modify_filter(self, files):
        self.target_files = files

        if files:
            self.file_filter.match = MatchAny(any=self.target_files)
            self.apply_file_filter = True
        else:
            self.apply_file_filter = False

    def assemble_filter(self):
        conditions = []
        if self.apply_file_filter:
            conditions.append(self.file_filter)
        if self.apply_chunk_validity_filter:
            conditions.append(self.chunk_validity_filter)

        return Filter(must=conditions)
    
    def set_response_limit(self, response_limit):
        self.response_limit = response_limit


def get_similarities(client, model, text, query_filter,
                     collection_name="kva_test_collection", response_limit = 5):
    """
    Searches for similar documents in a collection based on a given text and query filter.

    Args:
    - text (str): The text to search for similar documents.
    - query_filter (dict): A dictionary specifying the filter criteria for the search.
    - collection_name (str, optional): The name of the collection to search within, defaults to "kva_test_collection".
    - response_limit (int, optional): The maximum number of search results to return, defaults to 5.

    Returns:
    - dict: A dictionary containing lists of response texts, response types, scores, filenames, and page numbers.
    """

    search_result = client.search(
        collection_name=collection_name,
        query_vector=list(model.encode(text, normalize_embeddings=True).astype(float)),
        query_filter=query_filter,
        limit=response_limit,
        timeout=100)
    
    result_dict = defaultdict(list)
    
    for point in search_result:
        if not point.payload:
            continue
        result_dict['response_text'].append(point.payload["text"])
        result_dict['response_type'].append(point.payload["content_type"])
        result_dict['score'].append(point.score)
        result_dict['filename'].append(point.payload["filename"])
        result_dict['page_no'].append(point.payload["page_number"])

    return result_dict


def chat_view():

    client_host = config["dbs"]["qdrant"]["host"]
    client_port = config["dbs"]["qdrant"]["port"]

    embedding_model = config["embeddings"]["embedding_model"]
    collection_test = config["dbs"]["qdrant"]["collection_name"]

    chatter = QdrantChat(embedding_model, collection_test)
    chatter.connect(client_host, client_port)

    pn.extension("perspective")

    multi_select = pn.widgets.MultiSelect(name='Dokumendid', value=[],
    options=['jp3_09.pdf', 'AJP-3.2_EDA_V1_E_2288.pdf', 'fm3-09.pdf', 'app-6-c.pdf'], size=8)

    limit_slider = pn.widgets.EditableIntSlider(name='Vastuste arv', start=1, end=20, step=1, value=5)

    button = pn.widgets.Button(name='Rakenda filtrid', button_type='primary')

    def select_files(event):
        chatter.modify_filter(multi_select.value)
        chatter.set_response_limit(limit_slider.value)

    button.on_click(select_files)
    
    ci = ChatInterface(
        callback_exception='verbose',
        callback=chatter.chat_callback,
        widgets=pn.widgets.TextAreaInput(
            placeholder="Otsi dokumendist", auto_grow=True, max_rows=1
        ),
        user="Terminoloog", callback_user="Assistent",
        show_stop=False,
        show_rerun=False,
        show_undo=False,
    )

    filter_column = pn.Column(multi_select, limit_slider, button)
    layout = pn.Row(ci, filter_column)

    return layout
