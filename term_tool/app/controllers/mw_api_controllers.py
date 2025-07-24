import requests
from dotenv import load_dotenv
import os

import logging


logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

def get_data(query, dictionary_type):
    ref = 'collegiate'

    if dictionary_type == 'collegiate':
        api_key = os.getenv('MW_DICTIONARY')
    elif dictionary_type == 'thesaurus':
        api_key = os.getenv('MW_THESAURUS')
    else:
        return None

    url = f"https://dictionaryapi.com/api/v3/references/{dictionary_type}/json/{query}?key={api_key}"
    
    logger.info(f'Merriam Webster request URL: {url}')

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code