import requests
from dotenv import load_dotenv
import os
import json


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


def set_headers():

    api_key = os.getenv('COLLINS_PASSWORD')

    headers = {
        'Accept': 'application/json',
        'accessKey': api_key,
    }
    return headers


def get_dictionaries(headers):
    url = 'https://api.collinsdictionary.com/api/v1/dictionaries'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        dictionaries = response.json()
        return dictionaries
    except Exception as e:
        print(f"Failed to retrieve dictionaries: {e}")
        return []


def get_search_results(dict_code, search_word, page_size, page_index):
    headers = set_headers()

    url = f'https://api.collinsdictionary.com/api/v1/dictionaries/{dict_code}/search/?q={search_word}&pagesize={page_size}&pageindex={page_index}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to retrieve search results: {e}")
        return {}


def get_first_matching_entry(headers, dict_code, search_word):
    url = f'https://api.collinsdictionary.com/api/v1/dictionaries/{dict_code}/search/first?q={search_word}&format=xml'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to retrieve first/best mathicng entry: {e}")
        return {}
    

def get_entries_urls(results):
    urls = []
    for r in results.get('results', []):
        urls.append(r.get('entryUrl'))
    return urls


def get_entry_by_entry_url(entry_url):
    headers = set_headers()
    entry_url = entry_url.replace('http', 'https')
    try:
        response = requests.get(entry_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to retrieve entry: {e}")
        return None
    

def get_did_you_mean(headers, dict_code, search_word, entry_number):
    url = f'https://api.collinsdictionary.com/api/v1/dictionaries/{dict_code}/search/didyoumean?q={search_word}&entrynumber={entry_number}'
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        dictionaries = response.json()
        return dictionaries
    except Exception as e:
        print(f"Failed to retrieve suggestions: {e}")
        return []