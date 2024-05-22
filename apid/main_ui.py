from bokeh.sampledata.autompg import autompg
import panel as pn
import pandas as pd
from datetime import datetime
from collins.helpers import entry_data_to_dataframe, entry_cobuild_data_to_dataframe
from helpers import dataframes_to_excel
from iate.helpers import search_results_to_dataframe
from mw.helpers import json_to_df_with_definitions_and_usages

pn.extension('tabulator')


# Define a CSS style for text wrapping
css = """
.tabulator-cell {
    white-space: pre-wrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
"""
pn.config.raw_css.append(css)

def fetch_results(query, source_language, target_languages, num_pages, optional_parameters):
    iate_results = search_results_to_dataframe(query, source_language, target_languages, num_pages, optional_parameters)
    collins_english_results = entry_data_to_dataframe('english', query, 100, 1)
    collins_cobuild_advanced_british_results = entry_cobuild_data_to_dataframe('english-learner', query, 100, 1)
    collins_cobuild_advanced_american_results = entry_cobuild_data_to_dataframe('american-learner', query, 100, 1)
    mw_dict_results = json_to_df_with_definitions_and_usages(query)
    
    return iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results

def save_to_excel(dfs, filename):
    with pd.ExcelWriter(filename) as writer:
        for df, sheet_name in zip(dfs, ["IATE", "Collins English", "Collins British", "Collins American", "MW Dictionary"]):
            df.to_excel(writer, sheet_name=sheet_name, index=False)


query_input = pn.widgets.TextInput(name='Query', placeholder='Enter search query here...', value='warfare')
source_language_input = pn.widgets.TextInput(name='Source Language', placeholder='Enter source language here...', value='en')
target_languages_input = pn.widgets.TextInput(name='Target Languages', placeholder='Enter target languages here...', value='en, fr')
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
query_operator_input = pn.widgets.RadioBoxGroup(
    name='Query Operator',
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

def on_click(event):
    query = query_input.value
    source_language = source_language_input.value
    target_languages = target_languages_input.value.split(',')
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

    iate_results, collins_english_results, collins_cobuild_advanced_british_results, collins_cobuild_advanced_american_results, mw_dict_results = fetch_results(
        query, source_language, target_languages, num_pages, optional_parameters)
    
    response_area.clear()
    response_area.append(pn.pane.Markdown("### IATE"))
    response_area.append(pn.widgets.Tabulator(iate_results, groupby=['IATE link'], show_index=False, sizing_mode='stretch_width', layout='fit_columns'))

    response_area.append(pn.pane.Markdown("### Collins English"))
    response_area.append(pn.widgets.Tabulator(collins_english_results, show_index=False, sizing_mode='stretch_width', layout='fit_columns', width=800))
    response_area.append(pn.pane.Markdown("### Collins COBUILD Advanced British"))
    response_area.append(pn.widgets.Tabulator(collins_cobuild_advanced_british_results, show_index=False, sizing_mode='stretch_width', layout='fit_columns', width=800))
    response_area.append(pn.pane.Markdown("### Collins COBUILD Advanced American"))
    response_area.append(pn.widgets.Tabulator(collins_cobuild_advanced_american_results, show_index=False, sizing_mode='stretch_width', layout='fit_columns', width=800))
    response_area.append(pn.pane.Markdown("### MW Dictionary"))
    response_area.append(pn.widgets.Tabulator(mw_dict_results, show_index=False, sizing_mode='stretch_width', layout='fit_columns', width=800))

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
