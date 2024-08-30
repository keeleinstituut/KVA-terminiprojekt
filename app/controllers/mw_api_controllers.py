import requests
from dotenv import load_dotenv
import os


def get_data(query, dictionary_type):
    ref = 'collegiate'

    if dictionary_type == 'collegiate':
        api_key = os.getenv('MW_DICTIONARY')
    elif dictionary_type == 'thesaurus':
        api_key = os.getenv('MW_THESAURUS')
    else:
        print('Vale sõnastikutüüp.')
        return None

    uri = f"https://dictionaryapi.com/api/v3/references/{dictionary_type}/json/{query}?key={api_key}"
    
    response = requests.get(uri)

    if response.status_code == 200:
        return response.json()
    else:
        print(response.status_code)
        return response.status_code