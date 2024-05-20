import requests
import json


def perform_single_search(access_token, query, source, targets, num_pages=1, session=None, **kwargs):
    """
    Perform a single search request to the IATE API using an optional requests session.

    :param access_token: The Bearer token for authorization.
    :param query: The query string to search for.
    :param source: The source language of the terminology data.
    :param targets: A list of target languages for the terminology data.
    :param session: An optional session object for making requests.
    :param kwargs: Additional optional parameters (expand, 
                                                    offset, 
                                                    limit, 
                                                    fields_set_name, 
                                                    search_in_fields, 
                                                    search_in_term_types, 
                                                    filter_by_domains, 
                                                    cascade_domains, 
                                                    query_operator, 
                                                    filter_by_entry_collection, 
                                                    filter_by_entry_institution_owner, 
                                                    filter_by_entry_primarity, 
                                                    filter_by_source_term_reliability, 
                                                    filter_by_target_term_reliability, etc.)
    """
    url = "https://iate.europa.eu/em-api/entries/_search"

    headers = {
        'Accept': 'application/vnd.iate.entry+json; version=2',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    limit = kwargs.pop('limit', 10)
    offset = kwargs.pop('offset', 0)

    all_results = []
    
    for _ in range(num_pages):
        search_request = {
            "query": query,
            "source": source,
            "targets": targets,
            "offset": offset,
            "limit": limit, 
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

    print('all resultsi pikkus: ' + str(len(all_results)))
    return all_results


def perform_multi_search(access_token, queries, source, targets, **kwargs):
    """
    Perform a search request to the IATE API.

    :param access_token: The Bearer token for authorization.
    :param queries: A list of query strings to search for.
    :param source: The source language of the terminology data.
    :param targets: A list of target languages for the terminology data.
    :param kwargs: Additional optional parameters (offset, limit, expand, query_operator, etc.)
    """
    url = "https://iate.europa.eu/em-api/entries/_msearch?fields_set_name=minimal"
    
    payload = []
    for query in queries:
        search_request = {
            "search_request": {
                "query": query,
                "source": source,
                "targets": targets,
                **{k: v for k, v in kwargs.items() if k not in ['expand', 'offset', 'limit']}
            },
            **{k: v for k, v in kwargs.items() if k in ['expand', 'offset', 'limit']}
        }
        payload.append(search_request)

    json_payload = json.dumps(payload)
    
    headers = {
        'Accept': 'application/vnd.iate.entry+json; version=2',
        'Content-Type': 'application/vnd.iate.entry-multi-search+json;version=1',
        'Authorization': f'Bearer {access_token}'
    }

    response = requests.post(url, headers=headers, data=json_payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Failed to perform search', 'details': response.text}