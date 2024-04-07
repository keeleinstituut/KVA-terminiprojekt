import requests
import json


def get_single_entity_by_href(access_token, href, session=None):

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    if session:
        response = session.get(href, headers=headers)
    else:
        response = requests.get(href, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code
    
    
def get_languages(access_token, expand=True, offset=None, limit=None, trans_lang=None):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    params = {
        'expand': 'true' if expand else 'false',
    }

    if offset is not None:
        params['offset'] = offset
    if limit is not None:
        params['limit'] = limit
    if trans_lang is not None:
        params['trans_lang'] = trans_lang

    response = requests.get('https://iate.europa.eu/em-api/inventories/_languages', headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code


def get_domains(access_token, session=None):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    url = 'https://iate.europa.eu/em-api/domains/_tree'

    if session:
        response = session.get(url, headers=headers)
    else:
        response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code

    
def get_term_types(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    response = requests.get('https://iate.europa.eu/em-api/inventories/_term-types', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code
    

def get_reliabilities(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    response = requests.get('https://iate.europa.eu/em-api/inventories/_reliabilities', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code
    

def get_query_operators(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    response = requests.get('https://iate.europa.eu/em-api/inventories/_query-operators', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code
    

def get_searchable_fields(access_token):
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    response = requests.get('https://iate.europa.eu/em-api/inventories/_searchable-fields', headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code