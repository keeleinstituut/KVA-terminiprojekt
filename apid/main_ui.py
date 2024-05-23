import panel as pn
import pandas as pd
from collins.helpers import entry_data_to_dataframe, entry_cobuild_data_to_dataframe
from iate.helpers import search_results_to_dataframe
from mw.helpers import json_to_df_with_definitions_and_usages
import time

pn.extension('tabulator')

css = """
.tabulator-cell {
    white-space: pre-wrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
"""
pn.config.raw_css.append(css)


def on_click(event):

    query = query_input.value

    source_language = source_language_input.value
    target_languages = target_languages_input.value

    num_pages = num_pages_input.value
    search_in_fields = search_in_fields_input.value
    query_operator = query_operator_input.value

    if not query or not source_language or not target_languages or not search_in_fields:
        response_area.append(pn.pane.Markdown("**Error**: All fields must be filled out"))
        return

    optional_parameters = {
        'query_operator': query_operator,
        'search_in_fields': search_in_fields
    }
    
    fetch_algus = time.time()

    iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results = fetch_results(
        query, source_language, target_languages, num_pages, optional_parameters)
    
    fetch_lopp = time.time()

    print(f'Fetching results took {fetch_lopp - fetch_algus:.2f} seconds')

    kuva_algus = time.time()

    response_area.clear()
    response_area.append(pn.pane.Markdown("## IATE"))

    html_columns = {
        'IATE link': {'type': 'html'},
        'Termini allikaviide': {'type': 'html'},
        'Termini märkus': {'type': 'html'},
        'Termini märkuse allikaviide': {'type': 'html'},
        'Definitsioon': {'type': 'html'},
        'Definitsiooni allikaviited': {'type': 'html'},
        'Mõiste märkus': {'type': 'html'},
        'Mõiste märkuse allikaviide': {'type': 'html'},
        'Kasutusnäide': {'type': 'html'},
        'Kasutusnäite allikaviide': {'type': 'html'}
    }

    response_area.append(pn.widgets.Tabulator(iate_results, groupby=['IATE link'], show_index=False, formatters=html_columns,  layout='fit_columns', width=2000))

    response_area.append(pn.pane.Markdown("## Dictionaries"))

    combined_df = pd.concat([
        collins_english_results, 
        collins_cobuild_advanced_british_results, 
        collins_cobuild_advanced_american_results, 
        mw_dict_results
    ], ignore_index=True)

    response_area.append(pn.widgets.Tabulator(combined_df, show_index=False, layout='fit_columns', width=2000))

    kuva_lopp = time.time()

    print(f'Kuvamine võttis aega: {kuva_lopp - kuva_algus:.2f}')


def fetch_results(query, source_language, target_languages, num_pages, optional_parameters):
    iate_algus = time.time()
    iate_results = search_results_to_dataframe(query, source_language, target_languages, num_pages, optional_parameters)
    iate_lopp = time.time()

    print(f'Fetching IATE results took {iate_lopp - iate_algus:.2f} seconds')

    print('')

    collins_algus = time.time()
    collins_english_results = entry_data_to_dataframe('english', query, 100, 1)
    collins_lopp = time.time()
    print(f'Fetching Collins English results took {collins_lopp - collins_algus:.2f} seconds')

    collins_algus_2 = time.time()
    collins_cobuild_advanced_british_results = entry_cobuild_data_to_dataframe('english-learner', query, 100, 1)
    collins_lopp_2 = time.time()
    print(f'Fetching Collins Adv Br results took {collins_lopp_2 - collins_algus_2:.2f} seconds')

    collins_cob_am_algus = time.time()
    collins_cobuild_advanced_american_results = entry_cobuild_data_to_dataframe('american-learner', query, 100, 1)
    collins_cob_am_lopp = time.time()
    print(f'Fetching Collins Cobuild American results took {collins_cob_am_lopp - collins_cob_am_algus:.2f} seconds')

    mw_algus = time.time()
    mw_dict_results = json_to_df_with_definitions_and_usages(query)
    mw_lopp = time.time()
    print(f'Fetching MW results took {mw_lopp - mw_algus:.2f} seconds')

    return iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results


query_input = pn.widgets.TextInput(name='Query', placeholder='Enter search query here...', value='warfare')

source_language_input = pn.widgets.Select(
    name='Source Language', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value='en'
)

target_languages_input = pn.widgets.MultiChoice(
    name='Target Languages', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value=['et']
)

num_pages_input = pn.widgets.IntInput(name='Number of Pages', value=1, step=1)

search_in_fields_input = pn.widgets.CheckBoxGroup(
    name='Search In Fields',
    value=[0],
    options={
        'Term entry def': 0,
        'Term entry note': 2,
        'Term entry context': 3,
        'Language entry note': 7,
        'Entry code': 8,
        'Entry id': 9
    }
)

query_operator_input = pn.widgets.Select(
    value=5,
    options={
        "Any Word": 0,
        "All Words": 1,
        "Exact String": 2,
        "Exact Match": 3,
        "Regular Expression": 4,
        "Partial String": 5,
        "In": 6,
        "Not In": 7,
        "All": 8,
        "Is Empty": 9
    }
)

fetch_button = pn.widgets.Button(name='Search', button_type='primary')

response_area = pn.Column()

fetch_button.on_click(on_click)

input_widgets = pn.WidgetBox(
    query_input,
    source_language_input,
    target_languages_input,
    num_pages_input,
    pn.pane.Markdown("**Search In Fields**"),
    search_in_fields_input,
    pn.pane.Markdown("**Query Operator**"),
    query_operator_input,
    fetch_button
)

template = pn.template.BootstrapTemplate(
    title='Search from IATE and dictionaries',
    sidebar=[input_widgets],
)

template.main.append(response_area)

template.servable()