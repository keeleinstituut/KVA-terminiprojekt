import panel as pn
import pandas as pd
from utils.collins_api_helpers import entry_data_to_dataframe, entry_cobuild_data_to_dataframe
from utils.iate_api_helpers import search_results_to_dataframe
from utils.mw_api_helpers import json_to_df_with_definitions_and_usages
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
    #num_pages = num_pages_input.value
    num_pages=1
    search_in_fields = search_in_fields_input.value
    query_operator = query_operator_input.value

    if not query or not source_language or not target_languages or not search_in_fields:
        response_area.append(pn.pane.Markdown("**Viga**: Kõik väljad peavad olema täidetud"))
        return

    optional_parameters = {
        'query_operator': query_operator,
        'search_in_fields': search_in_fields
    }

    iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results = fetch_results(
        query, source_language, target_languages, num_pages, optional_parameters)
        
    response_area.clear()

    html_columns = {
        'Link': {'type': 'html'},
        'Termini allikas': {'type': 'html'},
        'Termini märkus': {'type': 'html'},
        #'Term note ref': {'type': 'html'},
        'Definitsioon': {'type': 'html'},
        #'Def ref': {'type': 'html'},
        'Märkus': {'type': 'html'},
        #'Note ref': {'type': 'html'},
        'Kontekst': {'type': 'html'},
        #'Context ref': {'type': 'html'}
    }

    iate_tab = pn.Column(
        pn.widgets.Tabulator(iate_results, groupby=['Link'], show_index=False, formatters=html_columns, layout='fit_columns', width=2000),
        margin=(20, 0)

    )

    combined_df = pd.concat([
        collins_english_results, 
        collins_cobuild_advanced_british_results, 
        collins_cobuild_advanced_american_results, 
        mw_dict_results
    ], ignore_index=True)

    dict_html_columns = {
        'Keelend': {'type': 'html'},
    }

    dictionaries_tab = pn.Column(
        pn.widgets.Tabulator(combined_df, formatters=dict_html_columns, show_index=False, layout='fit_columns', width=2000),
        margin=(20, 0)
    )

    tabs = pn.Tabs(
        ("IATE", iate_tab),
        ("Sõnaraamatud", dictionaries_tab)
    )

    response_area.append(tabs)


def fetch_results(query, source_language, target_languages, num_pages, optional_parameters):
    iate_results = search_results_to_dataframe(query, source_language, target_languages, num_pages, optional_parameters)

    collins_english_results = entry_data_to_dataframe('english', query, 100, 1)

    collins_cobuild_advanced_british_results = entry_cobuild_data_to_dataframe('english-learner', query, 100, 1)

    collins_cobuild_advanced_american_results = entry_cobuild_data_to_dataframe('american-learner', query, 100, 1)

    mw_dict_results = json_to_df_with_definitions_and_usages(query)

    return iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results

query_input = pn.widgets.TextInput(name='Otsisõna', placeholder='Trüki otsisõna siia', value='warfare', width=200)

source_language_input = pn.widgets.Select(
    name='Lähtekeel', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value='en',
    width=120
)

target_languages_input = pn.widgets.MultiChoice(
    name='Sihtkeeled', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value=['et'],
    width=200
)

#num_pages_input = pn.widgets.IntInput(name='Tulemuste lehekülgi', value=1, step=1, width=80)
search_in_fields_label = pn.pane.Markdown("**Otsi väljadelt**", width=200)

search_in_fields_input = pn.widgets.CheckBoxGroup(
    name='Otsi väljadelt',
    value=[0],
    options={
        'Termin': 0,
        'Termini märkus': 2,
        'Termini kasutusnäide': 3,
        'Keele tasandi märkus': 7
        #'Entry code': 8,
        #'Entry id': 9
    }
)

query_operator_input = pn.widgets.Select(
    value=5,
    name='Otsingu täpsus',
    options={
        #"Any Word": 0,
        "Kõik sõnad": 1,
        #"Exact String": 2,
        "Täpne vaste": 3,
        #"Regular Expression": 4,
        "Osaline vaste": 5
        #"In": 6,
        #"Not In": 7,
        #"All": 8,
        #"Is Empty": 9
    },
    width=150
)

fetch_button = pn.widgets.Button(name='Otsi', button_type='primary', width=100)

response_area = pn.Column()

fetch_button.on_click(on_click)

input_widgets = pn.WidgetBox(
    query_input,
    source_language_input,
    target_languages_input,
    search_in_fields_label,
    search_in_fields_input,
    query_operator_input,
    #num_pages_input,
    fetch_button,
    sizing_mode='stretch_width'
)

collapsible_input = pn.Card(input_widgets, title='Otsing', collapsible=True, collapsed=False, margin=(20, 0))

def api_view():
    return pn.Column(collapsible_input, response_area)

api_view().servable()