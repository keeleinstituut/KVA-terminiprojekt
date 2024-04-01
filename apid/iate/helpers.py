import re
from . import authentication_requests
from . import catalogue_requests
from . import entries_requests
import json
import pandas as pd


def print_single_search_results(query, source_language, target_languages, optional_parameters):

    tokens = authentication_requests.get_iate_tokens()

    access_token = tokens['tokens'][0]['access_token']

    result = entries_requests.perform_single_search(access_token, query, source_language, target_languages, **optional_parameters)

    domains = catalogue_requests.get_domains(access_token)
    
    if result:
        if 'items' in result:
            for item in result['items']:

                entry = catalogue_requests.get_single_entity_by_href(access_token, item['self']['href'])

                print_two_columns('IATE link:', 'https://iate.europa.eu/entry/result/' + str(entry['id']))

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

                        if 'note' in lang_data:
                            print_two_columns("Märkus:", lang_data['note']['value'])

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


def format_usage_examples(usages):
    value_for_df = ''
    for u in usages:
        cleaned_ref = re.sub(r'<a href="[^"]*" target="_blank">|</a>', '', u['reference']['text'])
        cleaned_ref = re.sub(r'<time datetime.*\">', '', cleaned_ref)
        cleaned_ref = cleaned_ref.replace('</time>', '')

        value_for_df += u['context'] + ' (' + cleaned_ref + '); '
    
    value_for_df = value_for_df.strip('; ').replace('<b>', '').replace('</b>', '').replace('&gt;', '>').replace('<br>', '').replace('<div>', '').replace('</div>', '')


    return value_for_df

def format_definition_or_note(definition_or_note):
    cleaned_def_or_note = re.sub(r'<a href="[^"]*" target="_blank">|</a>', '', definition_or_note)
    cleaned_def_or_note = re.sub(r'<time datetime="[^"]*">|</time>', '', cleaned_def_or_note)
    cleaned_def_or_note = cleaned_def_or_note.replace('<br>', '').replace('<p>', '').replace('</p>', '')

    if '"ENTRY_TO_ENTRY_CONVERTER"' in cleaned_def_or_note:
        cleaned_def_or_note = re.sub(r'\[ <a href="([^"]+)"[^>]*>IATE:(\d+) \]', r'[https://iate.europa.eu/entry/result/\2/all]', cleaned_def_or_note)

    return cleaned_def_or_note


def search_results_to_dataframe(query, source_language, target_languages, optional_parameters):
    results_list = []

    tokens = authentication_requests.get_iate_tokens()
    access_token = tokens['tokens'][0]['access_token']
    result = entries_requests.perform_single_search(access_token, query, source_language, target_languages, **optional_parameters)
    domains = catalogue_requests.get_domains(access_token)

    if result and 'items' in result:
        for item in result['items']:
            entry = catalogue_requests.get_single_entity_by_href(access_token, item['self']['href'])
            
            domain_hierarchy = []
            for domain in entry['domains']:
                domain_hierarchy.append(" > ".join(get_domain_hierarchy_by_code(domains, domain['code'])))
            domain_hierarchy_str = "; ".join(domain_hierarchy)

            for tl in target_languages:
                if tl in entry['language']:
                    lang_data = entry['language'][tl]
                    term_entries = lang_data.get('term_entries', [])
                    for term_entry in term_entries:
                        creation_time = entry['metadata']['creation']['timestamp'].split('T')[0]
                        modification_time = entry['metadata']['modification']['timestamp'].split('T')[0]

                        entry_data = {
                            'IATE link': 'https://iate.europa.eu/entry/result/' + str(entry['id']),
                            'Lisatud': creation_time,
                            'Muudetud': modification_time,
                            #'Status': str(entry['metadata']['status']),
                            'Valdkond': domain_hierarchy_str,
                            'Keel': tl.upper(),
                            'Termin': term_entry['term_value'],
                            'Definitsioon': format_definition_or_note(lang_data.get('definition', '')),
                            'Märkus': format_definition_or_note(lang_data['note']['value']) if 'note' in lang_data else '',
                            'Kasutusnäide': format_usage_examples(term_entry.get('contexts', []))
                            }

                        results_list.append(entry_data)
                else:
                    creation_time = entry['metadata']['creation']['timestamp'].split('T')[0]
                    modification_time = entry['metadata']['modification']['timestamp'].split('T')[0]
                    
                    entry_data = {
                        'IATE link': 'https://iate.europa.eu/entry/result/' + str(entry['id']),
                        'Lisatud': creation_time,
                        'Muudetud': modification_time,
                        #'Status': str(entry['metadata']['status']),
                        'Valdkond': domain_hierarchy_str,
                        'Keel': tl.upper(),
                        'Termin': '',
                        'Definitsioon': '',
                        'Märkus': '',
                        'Kasutusnäide': ''
                    }
                    results_list.append(entry_data)

    return pd.DataFrame(results_list)