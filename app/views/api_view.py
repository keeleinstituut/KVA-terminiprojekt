import panel as pn
import pandas as pd
from utils.collins_api_helpers import entry_data_to_dataframe, entry_cobuild_data_to_dataframe
from utils.iate_api_helpers import search_results_to_dataframe
from utils.mw_api_helpers import json_to_df_with_definitions_and_usages
import time
import logging


logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

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
    logger.info('API search button clicked.')

    query = query_input.value
    source_language = source_language_input.value
    target_languages = target_languages_input.value
    search_in_fields = search_in_fields_input.value
    query_operator = query_operator_input.value
    only_first_batch_value = only_first_batch_checkbox.value

    if not query or not source_language or not target_languages or not search_in_fields:
        response_area.append(pn.pane.Markdown("**Viga**: Kõik väljad peavad olema täidetud"))
        return

    fetch_button.disabled = True

    optional_parameters = {
        'query_operator': query_operator,
        'search_in_fields': search_in_fields
    }

    iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results = fetch_results(
        query, source_language, target_languages, only_first_batch_value, optional_parameters)

    fetch_button.disabled = False

    response_area.clear()

    html_columns = {
        'Link': {'type': 'html'},
        'Termini allikas': {'type': 'html'},
        'Termini märkus': {'type': 'html'},
        'Definitsioon': {'type': 'html'},
        'Märkus': {'type': 'html'},
        'Kontekst': {'type': 'html'}
    }

    tabulator_editors = {
        'str': {'type': 'list', 'valuesLookup': True},
    }

    iate_tab = pn.Column(
        pn.widgets.Tabulator(iate_results, 
                             groupby=['Link'], 
                             show_index=False, 
                             formatters=html_columns, 
                             layout='fit_columns', 
                             widths={
                                     'Link': '75', 
                                     'Lisatud': '100', 
                                     'Muudetud': '100', 
                                     'Valdkond': '125', 
                                     'Keel': '50', 
                                     'Termin': '200', 
                                     'Termini allikas': '300', 
                                     'Termini märkus': '200',
                                     'Definitsioon': '400',
                                     'Märkus': '200',
                                     'Kontekst': '300'
                                     },
                             editors=tabulator_editors, 
                             header_filters=True),
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
        pn.widgets.Tabulator(combined_df, 
                             formatters=dict_html_columns, 
                             show_index=False, 
                             editors=tabulator_editors,
                             layout='fit_columns',
                             widths={'Allikas': '250', 'Keelend': '150', 'Definitsioon': '600', 'Lühike definitsioon': '500', 'Näide': '600'},
                             header_filters=True),
        margin=(20, 0)
    )

    tabs = pn.Tabs(
        ("IATE", iate_tab),
        ("Sõnaraamatud", dictionaries_tab)
    )

    response_area.append(tabs)

def fetch_results(query, source_language, target_languages, only_first_batch, optional_parameters):

    logger.info('Started fetching results from IATE and dictionaries.')

    try:
        iate_results = search_results_to_dataframe(query, source_language, target_languages, only_first_batch, optional_parameters)
    except Exception as e:
        logger.warning(f'Error fetching results from IATE: {e}')
        iate_results = pd.DataFrame()

    try:
        collins_english_results = entry_data_to_dataframe('english', query, 100, 1)
    except Exception as e:
        logger.warning(f'Error fetching results from Collins English: {e}')
        collins_english_results = pd.DataFrame()

    try:
        collins_cobuild_advanced_british_results = entry_cobuild_data_to_dataframe('english-learner', query, 100, 1)
    except Exception as e:
        logger.warning(f'Error fetching results from Collins Cobuild Advanced British: {e}')
        collins_cobuild_advanced_british_results = pd.DataFrame()

    try:
        collins_cobuild_advanced_american_results = entry_cobuild_data_to_dataframe('american-learner', query, 100, 1)
    except Exception as e:
        logger.warning(f'Error fetching results from Collins Cobuild Advanced American: {e}')
        collins_cobuild_advanced_american_results = pd.DataFrame()

    try:
        mw_dict_results = json_to_df_with_definitions_and_usages(query)
    except Exception as e:
        logger.warning(f'Error fetching results from Merriam Webster: {e}')
        mw_dict_results = pd.DataFrame()

    return iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results

query_input = pn.widgets.TextInput(name='Otsisõna', placeholder='Trüki otsisõna siia', value='reconnaissance', width=200)

source_language_input = pn.widgets.Select(
    name='Lähtekeel', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value='en',
    width=120
)

target_languages_input = pn.widgets.MultiChoice(
    name='Sihtkeeled', 
    options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
    value=['et', 'en'],
    width=200
)

search_in_fields_label = pn.pane.Markdown("**Otsi väljadelt**", width=200)

# External logged in users can search in: Term, Term note, Term in context, Language level note
#
# "Term Entry Def": 0,
# "Term Entry Note": 2,
# "Term Entry Context": 3,
# "Language Entry Note": 7

search_in_fields_input = pn.widgets.CheckBoxGroup(
    name='Otsi väljadelt',
    value=[0],
    options={
        'Termin': 0,
        'Termini märkus': 2,
        'Termini kasutusnäide': 3,
        'Keele tasandi märkus': 7
    }
)

# IATE Matching options: https://handbook.iate2.eu/iate-search/main-search/expanded-search/matching-options/
# ‘All words’ is the default option and retrieves terms which contain all of the words in the search field.
#
# ‘Exact match’ retrieves terms which match the query exactly. By default, search strings containing characters 
#   with diacritic marks will retrieve the diacritic characters and the base characters as exact matches.
#
# ‘Exact string’ retrieves results which contain the search query exactly as it is, with full words. Results 
#   may contain additional words before or after the search string.
#
# ‘Partial string’ retrieves results containing the search string within a longer string. The string doesn’t 
#   need to be full words. For example, searching for ‘book’ will retrieve ֥‘book’, but also ‘e-book’, ‘booking’, 
#   ‘notebook’, etc. (i.e. results which contain ‘book’ inside a longer string). This matching option is the 
#   only one possible when searching in fields other than the Term field.
#
# If you select ‘Any word’, results matching any one of the words in the search string will be displayed below 
# the more relevant results (e.g. if you search for ‘European Commission’ with ‘Any wordְ’ selected, 
# ‘European citizen’ would be displayed below the ‘all words’ results listed above).
#
# ‘Regular expression’ allows advanced users to run more complex searches using these patterns:
#    https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-regexp-query.html.
#
# "Any word": 0,
# "All words": 1,
# "Exact String": 2,
# "Exact Match": 3,
# "Regular Expression": 4,
# "Partial String": 5,
# "In": 6,
# "Not In": 7,
# "All": 8,
# "Is Empty": 9,
# "Is Not Empty": 10,
# "Gt": 11,
# "Gte": 12,
# "Lt": 13,
# "Lte": 14,
# "Equals": 15,
# "Between": 16,
# "Only": 17,
# "Light Any Word": 18,

query_operator_input = pn.widgets.Select(
    value=2,
    name='Otsingu täpsus',
    options={
        "Ükskõik mis sõna": 0,
        "Kõik sõnad": 1,
        "Täpselt sama sõna": 2,
        "Täpselt sama vaste": 3,
        "Osaline vaste": 5
    },
    width=150,
    description='Kõik sõnad - leiab vasted, mis sisaldavad kõiki otsitud sõnu\n '
                'Täpselt sama sõna - leiab vasted, mis sisaldavad otsisõna\n '
                'Täpselt sama vaste - leiab täpselt selle, mis otsisõnaks on (diakriitilisi märke ignoreeritakse)\n '
                'Osaline vaste - leiab vasted, milles sisaldub otsisõna või otsisõna on mõne teise sõna osa'
)

only_first_batch_checkbox = pn.widgets.Checkbox(name='Kuva esimesed 10 IATE vastet')

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
    only_first_batch_checkbox,
    fetch_button,
    sizing_mode='stretch_width'
)

collapsible_input = pn.Card(input_widgets, title='Otsing', collapsible=True, collapsed=False, margin=(20, 0))

def api_view():

    return pn.Column(collapsible_input, response_area)

api_view().servable()