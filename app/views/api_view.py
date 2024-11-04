import panel as pn
import pandas as pd
import param
from utils.collins_api_helpers import entry_data_to_dataframe, entry_cobuild_data_to_dataframe
from utils.iate_api_helpers import search_results_to_dataframe
from utils.mw_api_helpers import json_to_df_with_definitions_and_usages
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


def fetch_results(query, source_language, target_languages, only_first_batch, optional_parameters, domains):

    logger.info('Started fetching results from IATE and dictionaries.')

    try:
        iate_results = search_results_to_dataframe(query, source_language, target_languages, only_first_batch, optional_parameters, domains)
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


class APIViewWidgets(param.Parameterized):

    def __init__(self, app, **params) -> None:

        self.app = app
        self.query_input = pn.widgets.TextInput(name='Otsisõna', placeholder='Trüki otsisõna siia', value='reconnaissance', width=200)

        self.source_language_input = pn.widgets.Select(
            name='Lähtekeel', 
            options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
            value='en',
            width=120
        )

        self.target_languages_input = pn.widgets.MultiChoice(
            name='Sihtkeeled', 
            options=['en', 'fr', 'de', 'et', 'ru', 'fi'], 
            value=['et', 'en'],
            width=200
        )

        self.search_in_fields_label = pn.pane.Markdown("**Otsi väljadelt**", width=200)


        self.search_in_fields_input = pn.widgets.CheckBoxGroup(
            name='Otsi väljadelt',
            value=[0],
            options={
                'Termin': 0,
                'Termini märkus': 2,
                'Termini kasutusnäide': 3,
                'Keele tasandi märkus': 7
            }
        )


        self.query_operator_input = pn.widgets.Select(
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

        self.only_first_batch_checkbox = pn.widgets.Checkbox(name='Kuva esimesed 10 IATE vastet')

        self.fetch_button = pn.widgets.Button(name='Otsi', button_type='primary', width=100)

        self.response_area = pn.Column()

        super().__init__(**params)
    
    @param.depends('fetch_button.value', watch=True)
    def on_click(self):
        logger.info('API search button clicked.')

        query = self.query_input.value
        source_language = self.source_language_input.value
        target_languages = self.target_languages_input.value
        search_in_fields = self.search_in_fields_input.value
        query_operator = self.query_operator_input.value
        only_first_batch_value = self.only_first_batch_checkbox.value

        if not query or not source_language or not target_languages or not search_in_fields:
            self.response_area.append(pn.pane.Markdown("**Viga**: Kõik väljad peavad olema täidetud"))
            return

        self.fetch_button.disabled = True

        optional_parameters = {
            'query_operator': query_operator,
            'search_in_fields': search_in_fields
        }

        iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results = fetch_results(
            query, source_language, target_languages, only_first_batch_value, optional_parameters, self.app)

        self.fetch_button.disabled = False

        self.response_area.clear()

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

        self.response_area.append(tabs)


def api_view(app):

    widget_handler = APIViewWidgets(app)

    input_widgets = pn.WidgetBox(
        widget_handler.query_input,
        widget_handler.source_language_input,
        widget_handler.target_languages_input,
        widget_handler.search_in_fields_label,
        widget_handler.search_in_fields_input,
        widget_handler.query_operator_input,
        widget_handler.only_first_batch_checkbox,
        widget_handler.fetch_button,
        sizing_mode='stretch_width'
    )

    collapsible_input = pn.Card(input_widgets, title='Otsing', collapsible=True, collapsed=False, margin=(20, 0))

    return pn.Column(collapsible_input, widget_handler.response_area)