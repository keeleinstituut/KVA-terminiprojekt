from . import authentication_requests
from . import catalogue_requests
from . import entries_requests
import json


def print_single_search_results(query, source_language, target_languages, optional_parameters):

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = entries_requests.perform_single_search(access_token, query, source_language, target_languages, **optional_parameters)

    domains = catalogue_requests.get_domains(access_token)
    
    if result:
        if 'items' in result:
            for item in result['items']:

                entry = catalogue_requests.get_single_entity_by_href(access_token, item['self']['href'])

                print_two_columns("ID:", str(entry['id']))
                for domain in entry['domains']:

                    print_two_columns("Valdkond:", (" > ".join(get_domain_hierarchy_by_code(domains, domain['code']))))

                print_two_columns("Loomise aeg:", entry['metadata']['creation']['timestamp'])
                print_two_columns("Muutmise aeg:", entry['metadata']['modification']['timestamp'])
                print_two_columns("Olek:", str(entry['metadata']['status']))
                print("\n")

                for tl in target_languages:
                    if tl in entry['language']:
                        lang_data = entry['language'][tl]
                        print(f"{tl.upper()}:")
                        if 'definition' in lang_data:
                            print_two_columns("Definitsioon:", lang_data['definition'])

                        if 'definition_references' in lang_data:
                            for def_ref in lang_data['definition_references']:
                                print_two_columns("Definitisiooni allikaviide:", def_ref['text'])

                        if 'term_entries' in lang_data:
                            for term_entry in lang_data['term_entries']:
                                print("\n")
                                print_two_columns("Termin:", term_entry['term_value'])

                                if 'term_references' in term_entry:
                                    for term_reference in term_entry['term_references']:
                                        print_two_columns("Termini allikaviide:", term_reference['text'])

                                
                                if 'contexts' in term_entry:
                                    for context in term_entry['contexts']:
                                        print_two_columns("Termini kasutusnäide:", context['context'])
                                        if 'reference' in context:
                                            print_two_columns("Termini kasutusnäite allikaviide:", context['reference']['text'])
      
                    else:
                        print_two_columns(f"Selles keeles tulemusi pole:", tl)
                    print('\n')
                print('----------------------------------------')

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


def print_query_operators():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_query_operators(access_token)

    if result:
        #print(json.dumps(result, indent=4))
        if 'items' in result:
            for item in result['items']:
                href = item['meta']['href']
                lang = catalogue_requests.get_single_entity_by_href(access_token, href)
                print(lang['name'], lang['code'])
        else:
            print('Tulemusi pole.')
    else:
        print('Tulemusi pole.')


def print_searchable_fields():
    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_searchable_fields(access_token)

    if result:
        #print(json.dumps(result, indent=4))
        if 'items' in result:
            for item in result['items']:
                href = item['meta']['href']
                lang = catalogue_requests.get_single_entity_by_href(access_token, href)
                print(lang['name'], lang['code'])
        else:
            print('Tulemusi pole.')
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


def get_domain_name_by_code(data, domain_code):
    def search_domain(domains):
        for domain in domains:
            if domain['code'] == domain_code:
                return domain['name']
            
            if 'subdomains' in domain:
                name = search_domain(domain['subdomains'])
                if name:
                    return name
    
    return search_domain(data['items']) or domain_code


def get_domain_hierarchy_by_code(data, domain_code, hierarchy=None):
    if hierarchy is None:
        hierarchy = []
    
    for item in data['items']:
        if item['code'] == domain_code:
            return hierarchy + [item['name']]
        
        if 'subdomains' in item:
            subdomain_result = get_domain_hierarchy_by_code({'items': item['subdomains']}, domain_code, hierarchy + [item['name']])
            if subdomain_result:
                return subdomain_result
    
    return None


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

    if result:
        for item in result['items']:
            href = item['meta']['href']
            reliability = catalogue_requests.get_single_entity_by_href(access_token, href)
            print(reliability)
    else:
        print('Tulemusi pole.')


def create_reliabilites_list():

    reliabilites_list = []

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = catalogue_requests.get_reliabilities(access_token)

    if result:
        for item in result['items']:
            href = item['meta']['href']
            reliability = catalogue_requests.get_single_entity_by_href(access_token, href)
            reliabilites_list.append(reliability)
    else:
        print('Tulemusi pole.')

    return reliabilites_list


def print_two_columns(label, text, width=40):
    lines = text.split('\n')
    first_line = True
    for line in lines:
        if first_line:
            print(f"{label.ljust(width)}{line}")
            first_line = False
        else:
            print(f"{' '.ljust(width)}{line}")