import authentication_requests
import catalogue_requests
import json
import entries_requests


def print_single_search_results(query, source_language, target_languages):

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = entries_requests.perform_single_search(access_token, query, source_language, target_languages)

    if result:
        print(result)
        if 'items' in result:
            for item in result['items']:

                entry = catalogue_requests.get_single_entity_by_href(access_token, item['self']['href'])

                print('Otsing: ' + query)

                print('\nDefinitsioonid\n')

                print(source_language + ': ' + entry['language'][source_language]['definition'])
                
                for tl in target_languages:
                    print(tl + ': ' + entry['language'][tl]['definition'])
        else:
            print('Tulemusi pole.')
    else:
        print('Tulemusi pole.')


def print_languages():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_languages(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            lang = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(lang['name'], lang['code'], lang['is_official'])
    else:
        print('Tulemusi pole.')


def print_domains():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_domains(access_token)

    if result:
        for item in result['items']:
            print(item['code'], item['name'])
    else:
        print('Tulemusi pole.')


def print_term_types():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_term_types(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            term_type = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(term_type)
    else:
        print('Tulemusi pole.')


def print_reliabilities():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_reliabilities(access_token)
    reliabilities = json.dumps(result, indent=4)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            reliability = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(reliability)
    else:
        print('Tulemusi pole.')