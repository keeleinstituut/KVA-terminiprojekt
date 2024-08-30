import asyncio
import logging
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http.models import (FieldCondition, Filter, MatchAny,
                                       MatchValue)
from sentence_transformers import SentenceTransformer
import re
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# import weave

# Configure logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

class FilterFactory():
    """
    A class for creating and managing filters for the Qdrant search engine.

    apply_chunk_validity_filter = True
    apply_file_filter = False
    apply_document_validity_filter = False
    Attributes:
        apply_chunk_validity_filter (bool): Whether to apply a filter for chunk validity.
        apply_file_filter (bool): Whether to apply a filter for specific documents.
        apply_document_validity_filter (bool): Whether to apply a filter for document validity.
        target_files (list): A list of target file names for the file_filter.
        response_limit (int): The maximum number of search results to return.
        chunk_validity_filter (FieldCondition): A filter condition for chunk validity.
        document_validity_filter (FieldCondition): A filter condition for filtering out outdated documents.
        file_filter (FieldCondition): A filter condition for querying only specific files.
    """

    apply_chunk_validity_filter = True
    apply_file_filter = False
    apply_document_validity_filter = False

    target_files = []
    response_limit = 5

    chunk_validity_filter = FieldCondition(
            key="validated",
            match=MatchValue(
                value=True,
            ),)
    
    document_validity_filter = FieldCondition(
    key = "is_valid",
    match=MatchValue(value=True,)
    )
    
    file_filter = FieldCondition(
        key="title",
        match=MatchAny(any=target_files),)


    def apply_filters(self, files: list, response_limit: int, document_validity: bool) -> None:
        """ Coordinates modfication of filter settings. """
        self.update_file_filter_target_files(files)
        self.set_response_limit(response_limit)
        self.set_document_validity(document_validity)
    
    def update_file_filter_target_files(self, files: list) -> None:
        """ Updates the target files for the file filter. """
        self.target_files = files
        if files:
            self.file_filter.match = MatchAny(any=self.target_files)
            self.apply_file_filter = True
        else:
            self.apply_file_filter = False
    
    def set_response_limit(self, response_limit: int) -> None:
        """ Sets the maximum number of search results to return. """
        self.response_limit = response_limit

    def set_document_validity(self, document_validity: bool) -> None:
        """ Sets whether to apply the document validity filter. """
        self.apply_document_validity_filter = document_validity

    # @weave.op()
    def assemble_filter(self) -> Filter:
        """ Assembles and returns the final filter based on the current settings. """
        conditions = []
        if self.apply_file_filter:
            conditions.append(self.file_filter)
        if self.apply_chunk_validity_filter:
            conditions.append(self.chunk_validity_filter)
        if self.apply_document_validity_filter:
            conditions.append(self.document_validity_filter)
        logger.info(f"Filter assembled: {Filter(must=conditions)}")
        return Filter(must=conditions)


class QdrantChat():
    """
    A class for interacting with a Qdrant vector search engine to retrieve relevant responses based on user input.

    Attributes:
        model (SentenceTransformer): The pre-trained sentence transformer model for encoding text.
        collection_name (str): The name of the Qdrant collection to search.
        filterfactory (FilterFactory): An instance of the FilterFactory class for managing filters for search results.
        prompt (str): A prompt to prepend to the user input before encoding.
        host (str): The hostname or IP address of the Qdrant server.
        port (int): The port number of the Qdrant server.
        client (QdrantClient): The Qdrant client instance for interacting with the server.
    """
    
    def __init__(self, model_name: str, collection_name: str, filterfactory: FilterFactory, prompt: str) -> None:
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        self.filterfactory = filterfactory
        self.prompt = prompt

    def connect(self, host: str, port: int) -> None:
        """  Connects to the Qdrant server using the provided host and port. """
        self.host = host
        self.port = port
        self.client = QdrantClient(host, port=port)

    async def chat_callback(self, contents: str, user, instance) -> str:
        """ A callback function for handling user input and generating responses. """
        await asyncio.sleep(1.8)
        db_filter = self.filterfactory.assemble_filter()
        logger.info('Filter assembled')

        # Get similar documents
        results = self.get_similarities(self.prompt + contents, db_filter)

        # Assemble response string
        response = ''
        for result in zip(results['response_text'], results['filename'], results['page_no']):
            response += f'File: {result[1]}\nPage: {result[2]}\n\n{result[0]}\n{"*"*15}\n'

        return response


    def get_similarities(self, text: str, query_filter: Filter) -> dict:
        """
        Searches for similar documents in the collection based on the given text and query filter.

        Args:
            text (str): The text to search for similar documents.
            query_filter (Filter): A dictionary specifying the filter criteria for the search.

        Returns:
            dict: A dictionary containing lists of response texts, response types, scores, filenames, and page numbers.
        """

        logger.info('Searching for similar documents')
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=list(self.model.encode(text, normalize_embeddings=True).astype(float)),
            query_filter=query_filter,
            limit=self.filterfactory.response_limit,
            timeout=100)
        logger.info('Search results retieved')
        
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
    

class LLMChat():
    
    chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a terminologist compiling a terminology database. "
         "You are searching for important information about a keyword and have found key sections from different documents."
         " You have four further tasks:"
         "1) if paragraphs contain any definitions for the term, extract all of the definitions and add references;"
         "2) list closely related terms, such as synonyms, hyponyms, hypernyms abbreviations etc that are used in key sections. Refer to a document and page no;"
         "3) extract coherent paragraphs that could be in any way useful for compiling a term entry and better understanding of the usage of the term. Present them in your response along with the correct document title, page number;"
         "4) list terms, abbreviations, synonyms that could be useful for further research of the keyword."
         "Format your report as:\n"
         "**TERM OF INTEREST**\n\n"
         "**Definitions:**\n1.	definition 1 (document, page no)\n2.	definition 2 (document, page no)\n\n"
         "**Related terms:**\n1.	term 1 (relation type, document, page no)\n\n"
         "**Important context:**\n1.	context 1 (document, page no)\n\n"
         "**See also**\nTerm 1, term 2, term 3 ... "),
        ("human", "Keyword: {user_input}"),
        ("human", "Key sections:\n{retrieval_results}")
    ]
    )

    def __init__(self, model_name: str, qdrant_chatter: QdrantChat, api_key) -> None:
        self.qdrant_chatter = qdrant_chatter
        self.model_name = model_name
        self.temperature = 0
        self.max_retries = 2
        self.api_key = api_key
        self.llm = self.connect_language_model()

    def connect_language_model(self):

        if re.match('claude', self.model_name):
            llm = ChatAnthropic(
                model_name=self.model_name,
                temperature=self.temperature,
                stop=None,
                base_url='',
                api_key=None,
                timeout=None,
                max_retries=self.max_retries,
                )
            
        elif re.match('gpt', self.model_name):
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=None,
                timeout=None,
                max_retries=self.max_retries,
                api_key=self.api_key
                )
            
        return llm
        
    # @weave.op()
    async def chat_callback(self, contents: str, user, instance) -> str:
        """ A callback function for handling user input and generating responses. """
        await asyncio.sleep(1.8)
        context = await self.qdrant_chatter.chat_callback(contents, '', '')
        prompt = self.chat_template.format_messages(user_input={contents}, retrieval_results={context})
        response = self.llm.invoke(prompt)
        return response.content   