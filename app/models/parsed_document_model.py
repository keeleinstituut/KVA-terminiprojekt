import math
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple
from utils.upload_helpers import divide_chunks

@dataclass
class Chunk:
    """
    A data class representing a chunk of text with additional metadata.

    Attributes:
        text (str): The text content of the chunk.
        page_number (int): The page number where the chunk is located.
        content_type (str): The type of content the chunk represents (e.g., 'text', 'image').
        validated (bool): Indicates whether the chunk has been validated.

    Methods:
        get_text(): Returns the text content of the chunk.
        get_data(): Returns a dictionary containing all the attributes of the chunk.
    """
    text: str
    page_number: int
    content_type: str
    validated: bool
    # Additional attributes as needed

    def get_text(self):
        """ Returns the text content of the chunk."""
        return self.text
    
    def get_data(self):
        """
        Returns a dictionary containing all the attributes of the chunk.

        Returns:
            dict: A dictionary with keys 'text', 'page_number', 'content_type', and 'validated',
                 and their corresponding values.
        """
        return {"text": self.get_text(),
                "page_number": self.page_number,
                "content_type": self.content_type,
                "validated": self.validated}


@dataclass
class DataField:
    """
    A data class representing a generic data field within a document.

    This class provides a framework for handling and processing various types of data fields
    extracted from documents, such as text blocks, footnotes, or tables. It includes methods
    for converting these data fields into structured chunks, validating the text within these chunks,
    and extracting specific information such as the page number where the data field is located.

    Attributes:
        document_field_json (List[dict]): A list of dictionaries representing the JSON structure of the document field.

    Methods:
        block_to_chunk(text: str, page_spans: dict, unit_block: list, valid_word_count: int = 10) -> Optional[Chunk]:
            Converts a precalculated text into a Chunk object, validating the text based on a minimum word count.

        _get_block_text(text: str, unit_block: list) -> str:
            Extracts the text of a block based on its start and end character indices.

        _get_block_page_number(page_spans: dict, block_start_index: int) -> int:
            Determines the page number where a block of text begins at.

        _validate_chunk_text(text, min_word_count=10) -> bool:
            Validates the text of a chunk based on a minimum word count.
    """

    document_field_json: List[dict]

    def block_to_chunk(self, text: str, page_spans: dict, unit_block: list, valid_word_count: int = 10) -> Optional[Chunk]:
        """
        Converts a block of text into a Chunk object, validating the text based on a minimum word count.

        Args:
            - text (str): The original text from which the block is extracted.
            - page_spans (dict): A dictionary mapping page numbers to their start and end character indices.
            - unit_block (list): A list of dictionaries representing the block, the 'start_char' attribute of the first element of the list is used. 
            - valid_word_count (int, optional): The minimum number of words required for the text to be considered valid. Defaults to 10.

        Returns:
            - Optional[Chunk]: A Chunk object if the block text is valid, None otherwise.
        """

        block_text = self._get_block_text(
            text, unit_block)

        block_start_index = unit_block[0]['start_char']

        # Finding page of the sentence block based on the beginning of the block
        page_number = self._get_block_page_number(
            page_spans, block_start_index)
        #print(block_start_index, page_number)

        if block_text.strip() != '':
            return Chunk(block_text, page_number,
                            'content_text', validated=self._validate_chunk_text(block_text, min_word_count=valid_word_count))
        return None

    def _get_block_text(self, text: str, unit_block: list) -> str:
        """
        Extracts the text of a block based on its start and end character indices.

        Args:
            - text (str): The original text from which the block is extracted.
            - unit_block (list): A list of dictionaries representing the block, including start and end character indices.

        Returns:
            str: The extracted text.
        """
        block_start_index = unit_block[0]['start_char']
        block_end_index = unit_block[-1]['end_char'] + 1

        block_text = text[block_start_index: block_end_index]
        return block_text

    def _get_block_page_number(self, page_spans: dict, block_start_index: int) -> int:
        """
        Determines the page number where the begginning of a block of text is located.

        Args:
            - page_spans (dict): A dictionary mapping page numbers to their start and end character indices.
            - block_start_index (int): The start character index of the block.

        Returns:
            int: The page number where the block is located.

        Raises:
            ValueError: If the block's start index does not fall within any of the provided page spans.
        """
        page_number = None
        for page_no, page_span in page_spans.items():
            if block_start_index in range(page_span[0], page_span[1]):
                page_number = page_no
                break
        if page_number is not None:
            return page_number
        raise ValueError('Sentence block is not in the provided page range.')
    
    def _validate_chunk_text(self, text, min_word_count=10):
        """
        Validates the text of a chunk based on a minimum word count.

        Args:
            - text (str): The text to be validated.
            - min_word_count (int, optional): The minimum number of words required for the text to be considered valid. Defaults to 10.

        Returns:
            bool: True if the text is valid (i.e., contains at least min_word_count words), False otherwise.
        """
        text = re.sub('(\n+| +)', ' ', text).strip()
        if len(text.split(' ')) < min_word_count - 1:
            return False
        return True


@dataclass
class TextualContent(DataField):

    def to_chunks_sentences(self, sentensizer, tokenizer, max_tokens: int = 512, n_sentences_in_block: int = 3, n_sentence_overlap: int = 0) -> List[Chunk]:
        """
        Converts the provided document data into a list of text chunks that are appropriate for further processing.

        Args:
            - sentensizer (object): An instance of a sentence boundary detection tool used to segment the text into sentences.
            - tokenizer (object): An instance of a tokenizer used to convert text into tokens.
            - max_tokens (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 512.
            - n_sentences_in_block (int, optional): The number of sentences that should be grouped together into a block before tokenization. Defaults to 3.
            - n_sentence_overlap (int, optional): The number of sentences that overlap between consecutive blocks. Defaults to 0.

        Returns:
            List[Chunk]: A list of Chunk objects, each representing a segment of text that has been processed according to the specified parameters. Each chunk adheres to the size restrictions set by max_tokens.
        """

        continuous_texts, continuous_text_page_spans = self.get_continuous_text_and_locations(self.document_field_json)

        # Chunking logic
        chunks = list()

        for i, text in enumerate(continuous_texts):
            page_spans = continuous_text_page_spans[i]
            sentence_data = sentensizer.get_sentences(text)

            print(f'{len(sentence_data)} sentences in file.')

            for sentence_block in divide_chunks(sentence_data, n_sentences_in_block):
                # todo: sentence overlap

                sentence_block_text = self._get_block_text(
                    text, sentence_block)
                sentence_block_tokens = tokenizer.get_tokens(
                    sentence_block_text)
                sentence_block_token_count = len(sentence_block_tokens)

                # If token limit is exceeded, split the text into equal parts by tokens
                if sentence_block_token_count > max_tokens:
                    new_block_size = round(sentence_block_token_count / math.ceil(
                        sentence_block_token_count / max_tokens))  # for more equal chunks
                    
                    for token_block in divide_chunks(sentence_block_tokens, new_block_size):
                        updated_token_block = list()
                        for d in token_block: # Changing token start_char global, instead sentence-based
                            d['start_char'] += sentence_block[0]['start_char']
                            d['end_char'] += sentence_block[0]['start_char']
                            updated_token_block.append(d)

                        extracted_chunk = self.block_to_chunk(
                            text, page_spans, updated_token_block)
                        if extracted_chunk:
                            chunks.append(extracted_chunk)

                # If token limit is not exceeded:
                else:
                    extracted_chunk = self.block_to_chunk(
                        text, page_spans, sentence_block)
                    if extracted_chunk:
                        chunks.append(extracted_chunk)

        return chunks


    def to_chunks(self, sentensizer, tokenizer, max_tokens: int = 200, n_sentences_in_block: int = 3, n_sentence_overlap: int = 0) -> List[Chunk]:
        """
        Converts the provided document data into a list of text chunks that are appropriate for further processing.

        Args:
            - sentensizer (object): An instance of a sentence boundary detection tool used to segment the text into sentences.
            - tokenizer (object): An instance of a tokenizer used to convert text into tokens.
            - max_tokens (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 512.
            - n_sentences_in_block (int, optional): The number of sentences that should be grouped together into a block before tokenization. Defaults to 3.
            - n_sentence_overlap (int, optional): The number of sentences that overlap between consecutive blocks. Defaults to 0.

        Returns:
            List[Chunk]: A list of Chunk objects, each representing a segment of text that has been processed according to the specified parameters. Each chunk adheres to the size restrictions set by max_tokens.
        """

        # 1) Full text extraction, finding page ranges
        chunks = list()
        full_text = ''
        page_spans = {}
        last_page_final_character_loc = -1

        for page_data in self.document_field_json:
            page_number = page_data['page_number']
            page_text = page_data['text'] + ' '
            page_text_length = len(page_text)
            full_text += page_text

            if page_number == 1:
                page_spans.update({1: (0, page_text_length)})
                last_page_final_character_loc = page_text_length
            else:
                page_spans.update({page_number: (last_page_final_character_loc, last_page_final_character_loc + page_text_length)})
                last_page_final_character_loc = last_page_final_character_loc + page_text_length

        # 2) Tokenizing full text
        tokenlist = tokenizer.get_tokens(full_text)

        # 2) Split tokens to blocks and generate Chunk objects
        for token_block in divide_chunks(tokenlist, max_tokens):
            token_block_text = full_text[token_block[0]['start_char']: token_block[-1]['end_char']+1]
            block_start_index  = token_block[0]['start_char']
            # Finding page of the sentence block based on the beginning of the block
            page_number = self._get_block_page_number(
                page_spans, block_start_index)

            if token_block_text.strip() != '':
                extracted_chunk =  Chunk(token_block_text, page_number,
                                'content_text', validated=self._validate_chunk_text(token_block_text, min_word_count=10))
            else:
                extracted_chunk = None

            if extracted_chunk:
                        chunks.append(extracted_chunk)

        return chunks

    def get_continuous_text_and_locations(self, document_field: List[dict]) -> Tuple[List[str], List[dict]]:
        """
        Extracts continuous text blocks from the input document data along with their respective page locations.

        This method processes a list of dictionaries representing pages of a document. Each dictionary must contain page-specific text and its corresponding page number. 
        The method concatenates text from consecutive pages and marks the span of text that each page covers in the concatenated result.

        Args:
            - document_field (List[dict]): A list of dictionaries, where each dictionary represents a page and contains keys 'page_number' and 'text'.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing two elements:
                1. A list of concatenated text blocks from the document's consecutive pages.
                2. A list of dictionaries, each dictionary mapping page numbers to the span of characters (start and end indices) that the page's text occupies in the concatenated text block.
        """

        continuous_text_page_spans = list()
        continuous_texts = list()
        last_page = -1
        last_page_final_character_loc = -1

        # Joining continuous texts and marking page spans
        for page_data in document_field:
            page_number = page_data['page_number']
            page_text = page_data['text'] + ' '
            page_text_length = len(page_data['text'])

            if last_page != page_number - 1:
                continuous_texts.append(page_text)
                continuous_text_page_spans.append({
                    page_number: (0, page_text_length +1)
                })
                last_page = page_number
                last_page_final_character_loc = page_text_length

            else:
                continuous_texts[-1] += page_text
                page_text_length += last_page_final_character_loc
                continuous_text_page_spans[-1].update({
                    page_number: (last_page_final_character_loc + 1,
                                  page_text_length + 2)
                })
                last_page = page_number
                last_page_final_character_loc = page_text_length + 1
        return continuous_texts, continuous_text_page_spans

@dataclass
class Document:
    """
    A data class representing a document with various metadata and references to other data fields.

    Attributes:
        json_filename (str): The filename of the JSON representation of the document.
        filename (str): The original filename of the document.
        publication (str): The name of the publication where the document was published.
        publication_year (int): The year of publication.
        title (str): The title of the document.
        author (str): The author of the document.
        languages (List[str]): A list of languages the document is available in.
        field_keywords (List[str]): A list of field-specific keywords associated with the document.

        content (TextualContent): Textual content of the document.


    Methods:
        get_metadata(): Returns a dictionary containing the document metadata that needs to be included in the database.

    """
    
    json_filename: str
    filename: str
    publication: str
    publication_year: int
    title: str
    author: str
    languages: List[str]
    field_keywords: List[str]
    is_valid: bool
    content: TextualContent 

    def get_metadata(self) -> dict:
        return {
            'json_filename': self.json_filename,
            'filename': self.filename,
            'publication': self.publication,
            'publication_year': self.publication_year,
            'title': self.title,
            'author': self.author,
            'languages': self.languages,
            'keywords': self.field_keywords,
            'is_valid': self.is_valid}

