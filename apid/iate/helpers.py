import auth_requests
import search_requests


def print_single_search_results(query, source_language, target_languages):

    tokens = auth_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = search_requests.perform_single_search(access_token, query, source_language, target_languages)

    #print(json.dumps(result, indent=4))

    for item in result['items']:
        #print(item['self']['href'])
        entry = search_requests.get_term_entry(access_token, item['self']['href'])

        print('Otsing: ' + query)

        print('\nDefinitsioonid\n')

        print(source_language + ': ' + entry['language'][source_language]['definition'])
        
        for tl in target_languages:
            print(tl + ': ' + entry['language'][tl]['definition'])