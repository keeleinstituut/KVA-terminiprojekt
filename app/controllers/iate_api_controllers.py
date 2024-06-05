import requests
import json
from .token_controller import TokenController

token_controller = TokenController()

def perform_single_search(access_token, query, source, targets, session=None, **kwargs):
    url = "https://iate.europa.eu/em-api/entries/_search"

    headers = {
        'Accept': 'application/vnd.iate.entry+json; version=2',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    all_results = []
    current_url = url

    while current_url:
        search_request = {
            "query": query,
            "source": source,
            "targets": targets,
            "fields_set_name": "minimal",
            **kwargs
        }

        json_payload = json.dumps(search_request)

        if session:
            response = session.post(current_url, headers=headers, data=json_payload)
        else:
            response = requests.post(current_url, headers=headers, data=json_payload)

        if response.status_code != 200:
            return {'error': 'Failed to perform search', 'details': response.text}
        
        data = response.json()

        items = data.get('items', [])

        if not items:
            break

        all_results.extend(items)

        #print(f"Batch size: {len(items)}, Total fetched so far: {len(all_results)}")

        next_link = data.get('next', {}).get('href')
        if not next_link:
            break

        current_url = next_link

    #print(f"Total size: {len(all_results)}")

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
