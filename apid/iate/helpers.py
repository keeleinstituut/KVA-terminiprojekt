import authentication_requests
import catalogue_requests
import json
import entries_requests


def print_single_search_results(query, source_language, target_languages):

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = entries_requests.perform_single_search(access_token, query, source_language, target_languages)

    concept = json.dumps(result, indent=4)

    if result:
        if 'items' in result:
            for item in result['items']:

                entry = catalogue_requests.get_single_entity_by_href(access_token, item['self']['href'])

                print(f"ID: {entry['id']}")
                for domain in entry['domains']:
                    print(f"Valdkonna kood: {domain['code']}")

                print(f"Loomise aeg: {entry['metadata']['creation']['timestamp']}")
                print(f"Muutmise aeg: {entry['metadata']['modification']['timestamp']}")
                print(f"Olek: {entry['metadata']['status']} \n")

                for tl in target_languages:
                    if tl in entry['language']:
                        lang_data = entry['language'][tl]
                        print(f"{tl.upper()}:")
                        print(f"Definitsioon: {lang_data['definition']}")

                        for def_ref in lang_data['definition_references']:
                            print(f"Definitisiooni allikaviide: {def_ref['text']}")

                        if 'term_entries' in lang_data:
                            for term_entry in lang_data['term_entries']:
                                print(f"\nTermin: {term_entry['term_value']}")

                                if 'term_references' in term_entry:
                                    for term_reference in term_entry['term_references']:
                                        print(f"Termini allikaviide: {term_reference['text']}")

                                
                                if 'contexts' in term_entry:
                                    for context in term_entry['contexts']:
                                        print(f"Termini kasutusnäide: {context['context']}")
                                        if 'reference' in context:
                                            print(f"Termini kasutusnäite allikaviide: {context['reference']}")

                                if 'metadata' in term_entry:
                                    print(f"Termini usaldusväärsus: {term_entry['metadata']['reliability']}")
                                                   
                    else:
                        print(f"Selles keeles tulemusi pole: {tl}")
                    print('\n')
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