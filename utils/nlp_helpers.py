import math
from typing import List

import spacy
from transformers import AutoTokenizer
from utils.upload_helpers import divide_chunks


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