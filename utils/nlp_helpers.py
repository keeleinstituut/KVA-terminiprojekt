from typing import List

from transformers import AutoTokenizer


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