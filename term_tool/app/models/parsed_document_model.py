from dataclasses import dataclass
import re
from typing import List
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

    def get_text(self):
        """Returns the text content of the chunk."""
        return self.text

    def get_data(self):
        """
        Returns a dictionary containing all the attributes of the chunk.

        Returns:
            dict: A dictionary with keys 'text', 'page_number', 'content_type', and 'validated',
                 and their corresponding values.
        """
        return {
            "text": self.get_text(),
            "page_number": self.page_number,
            "content_type": self.content_type,
            "validated": self.validated,
        }


@dataclass
class TextualContent:
    """
    A class for processing and chunking text content extracted from documents.

    Attributes:
        document_field_json (List[dict]): A list of dictionaries where each entry contains page-specific data,
                                          including page number and text content.
    """

    document_field_json: List[dict]

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
        raise ValueError("Block is not in the provided page range.")

    def _validate_chunk_text(self, text, min_word_count=10):
        """
        Validates the text of a chunk based on a minimum word count.

        Args:
            - text (str): The text to be validated.
            - min_word_count (int, optional): The minimum number of words required for the text to be considered valid. Defaults to 10.

        Returns:
            bool: True if the text is valid (i.e., contains at least min_word_count words), False otherwise.
        """
        text = re.sub("(\n+| +)", " ", text).strip()
        if len(text.split(" ")) < min_word_count - 1:
            return False
        return True

    def to_chunks(self, tokenizer, max_tokens: int = 200) -> List[Chunk]:
        """
        Converts the provided document data into a list of text chunks that are appropriate for further processing.

        Args:
            - tokenizer (object): An instance of a tokenizer used to convert text into tokens.
            - max_tokens (int, optional): The maximum number of tokens allowed in each chunk. Defaults to 200.

        Returns:
            List[Chunk]: A list of Chunk objects, each representing a segment of text that has been processed according to the specified parameters. Each chunk adheres to the size restrictions set by max_tokens.
        """

        # 1) Full text extraction, finding page ranges
        chunks = list()
        full_text = ""
        page_spans = {}
        last_page_final_character_loc = -1

        for page_data in self.document_field_json:
            page_number = page_data["page_number"]
            page_text = page_data["text"] + " "
            page_text_length = len(page_text)
            full_text += page_text

            if page_number == 1:
                page_spans.update({1: (0, page_text_length)})
                last_page_final_character_loc = page_text_length
            else:
                page_spans.update(
                    {
                        page_number: (
                            last_page_final_character_loc,
                            last_page_final_character_loc + page_text_length,
                        )
                    }
                )
                last_page_final_character_loc = (
                    last_page_final_character_loc + page_text_length
                )

        # 2) Tokenizing full text
        tokenlist = tokenizer.get_tokens(full_text)

        # 3) Split tokens to blocks and generate Chunk objects
        for token_block in divide_chunks(tokenlist, max_tokens):
            token_block_text = full_text[
                token_block[0]["start_char"] : token_block[-1]["end_char"] + 1
            ]
            block_start_index = token_block[0]["start_char"]
            # Finding page of the block based on the beginning of the block
            page_number = self._get_block_page_number(page_spans, block_start_index)

            if token_block_text.strip() != "":
                extracted_chunk = Chunk(
                    token_block_text,
                    page_number,
                    "content_text",
                    validated=self._validate_chunk_text(
                        token_block_text, min_word_count=10
                    ),
                )
            else:
                extracted_chunk = None

            if extracted_chunk:
                chunks.append(extracted_chunk)

        return chunks


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
        is_valid (bool): Is the document valid or outdated.
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
            "json_filename": self.json_filename,
            "filename": self.filename,
            "publication": self.publication,
            "publication_year": self.publication_year,
            "title": self.title,
            "author": self.author,
            "languages": self.languages,
            "keywords": self.field_keywords,
            "is_valid": self.is_valid,
        }
