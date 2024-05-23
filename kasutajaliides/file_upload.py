import collections
import param
import panel as pn
import json

BUTTON_WIDTH = 125

INITIAL_METADATA = {"filename" : "fname.pdf",
"publication": "", 
"publication_year": 0000,
"title": "{}",
"author": "",
"languages": [
"en"
],
"field_keywords": [
],
"header_height": 0,
"footer_height": 0,
"custom_regex": ""}

METADATA_FIELD_MAP = collections.OrderedDict(
    {"Publikatsioon": "publication",
     "Aasta": "publication_year",
     "Pealkiri": "title",
     "Autor": "author",
     "Keeled": "languages",
     "Märksõnad": "field_keywords"}
)


class Text(param.Parameterized):
    value = param.String(default='', allow_None=True, doc="The text value")

def update_json(**kwargs):
    """ Update INITIAL_METADATA if a new value is set to any of the metadata fields """
    print(kwargs)
    for key, value in kwargs.items():
        if not value:
            continue
        INITIAL_METADATA[key] = str(value).strip()
    return json.dumps(INITIAL_METADATA)

#pn.extension(sizing_mode="stretch_width", design="material")

pn.extension(design="material")    

# PDF browsing 
file_input = pn.widgets.FileInput(accept='.pdf')
pdf_viewer = pn.pane.PDF(file_input, width=500, height=800)

# Collecting input fields to bind to json updates
user_input_widgets = collections.OrderedDict({meta_field: pn.widgets.TextInput(name=display_field) for display_field, meta_field in METADATA_FIELD_MAP.items()})
user_input_widgets['filename'] = file_input.param.filename

update_json_bound = pn.bind(update_json, **user_input_widgets)

right_column = pn.Column(file_input, *user_input_widgets.values(), pn.pane.JSON(update_json_bound, name='JSON') )
left_column = pn.Column(pdf_viewer)

page_layout = pn.Row(left_column, right_column)

template = pn.template.VanillaTemplate(title="Faili üleslaadimine")
template.main.append(page_layout)

template.servable()



