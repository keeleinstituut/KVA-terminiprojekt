import math
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple


@dataclass
class Chunk:
    text: str
    page_number: int
    content_type: str
    validated: bool
    # Additional attributes as needed

    def get_text(self):
        return self.text
    
    def get_data(self):
        return {"text": self.get_text(),
                "page_number": self.page_number,
                "content_type": self.content_type,
                "validated": self.validated}


@dataclass
class DataField:

    document_field_json: List[dict]

    def block_to_chunk(self, text: str, page_spans: dict, unit_block: list) -> Optional[Chunk]:
        block_text = self._get_block_text(
            text, unit_block)
        block_start_index = unit_block[0]['start_char']

        # Finding page of the sentence block based on the beginning of the block
        page_number = self._get_block_page_number(
            page_spans, block_start_index)

        if block_text.strip() != '':
            return Chunk(block_text, page_number,
                            'content_text', validated=self._validate_chunk_text(text))
        return None

    def _get_block_text(self, text: str, unit_block: list) -> str:
        block_start_index = unit_block[0]['start_char']
        block_end_index = unit_block[-1]['end_char'] + 1

        block_text = text[block_start_index: block_end_index]
        return block_text

    def _get_block_page_number(self, page_spans: dict, block_start_index: int) -> int:
        page_number = None
        for page_no, page_span in page_spans.items():
            if block_start_index in range(page_span[0], page_span[1]):
                page_number = page_no
                break
        if page_number is not None:
            return page_number
        raise ValueError('Sentence block is not in the provided page range.')
    
    def _validate_chunk_text(self, text):
        text = re.sub('(\n+| +)', ' ', text).strip()
        if len(text.split(' ')) < 10:
            return False
        return True



@dataclass
class ContentTextData(DataField):

    def to_chunks(self, sentensizer, tokenizer, max_tokens: int = 512, n_sentences_in_block: int = 3, n_sentence_overlap: int = 0) -> List[Chunk]:

        continuous_texts, continuous_text_page_spans = self.get_continuous_text_and_locations(self.document_field_json)

        # Chunking logic
        chunks = list()

        for i, text in enumerate(continuous_texts):
            page_spans = continuous_text_page_spans[i]
            sentence_data = sentensizer.get_sentences(text)

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
                        extracted_chunk = self.block_to_chunk(
                            sentence_block_text, page_spans, token_block)
                        if extracted_chunk:
                            chunks.append(extracted_chunk)

                # If token limit is not exceeded:
                else:
                    extracted_chunk = self.block_to_chunk(
                        text, page_spans, sentence_block)
                    if extracted_chunk:
                        chunks.append(extracted_chunk)

        return chunks

    def get_continuous_text_and_locations(self, document_field: List[dict]) -> Tuple[List[str], List[dict]]:
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
                    page_number: (0, page_text_length)
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
class FootnoteData(DataField):

    def to_chunks(self) -> List[Chunk]:
        # Example processing: each text becomes a chunk
        chunks = list()
        for page_data in self.document_field_json:
            page_number = page_data['page_number']
            for footnote_block in page_data['texts']:
                for text in footnote_block:
                    chunks.append(Chunk(text, page_number, 'footnote',
                                        validated=self._validate_chunk_text(text)))
        return chunks

    def _validate_chunk_text(self, text):
        # Is there anything to validate on? Sentence length?
        return True


class TermData(DataField):

    def to_chunks(self) -> List[Chunk]:
        # Example processing: each text becomes a chunk
        chunks = list()
        for page_data in self.document_field_json:
            page_number = page_data['page_number']
            for text in page_data['texts']:
                chunks.append(Chunk(text, page_number, 'term',
                              validated=self._validate_chunk_text(text)))
        return chunks

    def _validate_chunk_text(self, text) -> bool:
        # todo: text should be longer than one token
        return True


@dataclass
class TableData(DataField):

    def to_chunks(self, tokenizer, max_tokens: int = 512) -> List[Chunk]:
        # todo parse one row as a context. Check validity and parse to sentences.
        chunks = list()
        for page_data in self.document_field_json:
            page_number = page_data['page_number']
            rows = list()

            for table in page_data['texts']:
                rows.append(' '.join(table['columns']))
                for table_row in table['data']:
                    table_row = ['' if el == None else el for el in table_row] # replace null values with a string
                    rows.append(' '.join(table_row))
            
            for row in rows:
                if row.strip() == '':
                    continue

                row_tokens = tokenizer.get_tokens(row)
                row_tokens_count = len(row_tokens)
                
                if row_tokens_count > max_tokens:
                    
                    new_block_size = round(row_tokens_count / math.ceil(
                        row_tokens_count / max_tokens))
                    
                    for token_block in divide_chunks(row_tokens, new_block_size):
                        extracted_chunk = self.block_to_chunk(
                            row, {page_number: (token_block[0]['start_char'],token_block[-1]['end_char'])}, token_block)
                        if extracted_chunk:
                            chunks.append(extracted_chunk)
                
                else:
                    chunks.append(Chunk(row, page_number, 'table',
                              validated=self._validate_chunk_text(row)))
                    
        return chunks


@dataclass
class Document:
    json_filename: str
    filename: str
    publication: str
    publication_year: int
    title: str
    author: str
    languages: List[str]
    field_keywords: List[str]
    header_height: int
    footer_height: int
    table_extraction_strategy: str
    horizontal_sorting: bool
    footnote_regex: str
    footnote_group: int
    custom_regex: dict

    # References to other data fields
    term_data: TermData
    content_text_data: ContentTextData 
    footnote_data: FootnoteData
    table_data: TableData

    def get_metadata(self) -> dict:
        return {
            'json_filename': self.json_filename,
            'filename': self.filename,
            'publication': self.publication,
            'publication_year': self.publication_year,
            'title': self.title,
            'author': self.author,
            'languages': self.languages,
            'keywords': self.field_keywords}


def divide_chunks(l, n):
    # Yield successive n-sized chunks from l.
    for i in range(0, len(l), n):
        yield l[i: i + n]

