import requests
import json
from .token_controller import TokenController

token_controller = TokenController()

def perform_single_search(access_token, query, source, targets, num_pages=5, session=None, **kwargs):
    url = "https://iate.europa.eu/em-api/entries/_search"

    headers = {
        'Accept': 'application/vnd.iate.entry+json; version=2',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    limit = kwargs.pop('limit', 100)
    offset = kwargs.pop('offset', 0)

    all_results = []

    for _ in range(num_pages):
        search_request = {
            "query": query,
            "source": source,
            "targets": targets,
            "offset": offset,
            "limit": limit,
            "fields_set_name": "minimal", 
            **kwargs           
        }

        json_payload = json.dumps(search_request)

        if session:
            response = session.post(url, headers=headers, data=json_payload)
        else:
            response = requests.post(url, headers=headers, data=json_payload)

        if response.status_code != 200:
            return {'error': 'Failed to perform search', 'details': response.text}
        
        data = response.json()

        items = data.get('items', [])

        if not items:
            break

        all_results.extend(items)

        if len(items) < limit:
            break

        offset += limit

    return all_results


def get_single_entity_by_href(access_token, href, session):
    access_token = token_controller.get_access_token()
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
