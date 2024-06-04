import collections
import param
import panel as pn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.controllers.pdf_upload_controller import upload_to_db
from utils.upload_helpers import normalized_keywords



BUTTON_WIDTH = 125

USER_INPUT_FIELDS = collections.OrderedDict(
    {"publication": pn.widgets.TextInput(name='Publikatsioon'),
     "publication_year": pn.widgets.IntInput(name='Aasta', value=2024, start=0, end=3000),
     "title": pn.widgets.TextInput(name='Pealkiri'),
     "author": pn.widgets.TextInput(name='Autor'),
     "languages": pn.widgets.TextInput(name='Keeled', description='Eralda keeled komadega'),
     "keywords": pn.widgets.TextInput(name='Märksõnad', description='Eralda märksõnad komadega'),
     "is_valid": pn.widgets.Checkbox(name='Kehtetu')}
)

DEFAULT_METADATA = {"publication": "",
              "publication_year": 2024,
              "title": "",
              "author": "",
              "languages": [],
              "keywords": [],
              "is_valid": True}

class DataHandler(param.Parameterized):

    action = param.Action(
        default=lambda x: x.param.trigger('action'), label='Salvesta')

    user_input_widgets = collections.OrderedDict()

    # PDF browsing
    pdf_input = pn.widgets.FileInput(accept='.pdf', name='Vali fail', filename='')
    pdf_viewer = pn.pane.PDF(pdf_input, width=500, height=800)

    submit_button = pn.widgets.Button(button_type='primary', name='Salvesta', align='start', width=BUTTON_WIDTH)
    new_entry_button = pn.widgets.Button(button_type='primary', name='Uus fail', align='end', width=BUTTON_WIDTH)

    metadata = DEFAULT_METADATA

    def __init__(self,**params):
        super().__init__(**params)
        self.user_input_widgets['filename'] = pn.widgets.TextInput(name='Faili pealkiri', placeholder=self.pdf_input.param.filename, disabled=True)
        self.user_input_widgets.update({meta_field: widget 
                                for meta_field, widget in USER_INPUT_FIELDS.items()})

    @param.depends('submit_button.value')
    def refresh_metadata(self):
        for name, widget in self.user_input_widgets.items():
            if name == 'is_valid':
                self.metadata[name] = not widget.value
            if name in ['keywords', 'languages']:
                self.metadata[name] = normalized_keywords(widget.value)
            if name == 'filename':
                self.metadata[name] = widget.placeholder
            elif isinstance(widget.value, str):
                self.metadata[name] = widget.value.strip()
            else:
                self.metadata[name] = widget.value
        
        if self.pdf_input.value:
            print('Loading:\n', self.metadata)
            self.submit_button.disabled=True
            upload_to_db(self.pdf_input.value, self.metadata)
            
    @param.depends('new_entry_button.value')
    def new_input(self):
        """
        Resetting fields to upload a new file.
        """
        for name, widget in self.user_input_widgets.items():
            if isinstance(widget, pn.widgets.TextInput):
                widget.value = ""
            else:
                widget.value = DEFAULT_METADATA[name]
        if self.pdf_input.value:
            self.pdf_input.value = None
            self.pdf_input.filename = ''

        self.metadata = DEFAULT_METADATA
        self.submit_button.disabled=False

def file_upload():

    metadata = DataHandler()

    right_column = pn.Column(metadata.pdf_viewer)
    left_column = pn.Column(metadata.pdf_input,
                            *metadata.user_input_widgets.values(),
                            pn.Row(metadata.submit_button, metadata.refresh_metadata, 
                                    metadata.new_entry_button, metadata.new_input)
                            )

    page_layout = pn.Row(left_column, right_column)

    return page_layout