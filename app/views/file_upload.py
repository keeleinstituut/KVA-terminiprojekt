import collections
import param
import panel as pn

from app.controllers.pdf_upload_controller import upload_to_db
from utils.upload_helpers import normalized_input_lists
import time
import copy

BUTTON_WIDTH = 125
INPUT_WIDTH = 300

USER_INPUT_FIELDS = collections.OrderedDict(
    {"publication": pn.widgets.TextInput(name='Publikatsioon'),
     "publication_year": pn.widgets.IntInput(name='Aasta', value=2024, start=0, end=3000),
     "title": pn.widgets.TextInput(name='Pealkiri'),
     "author": pn.widgets.TextInput(name='Autor'),
     "languages": pn.widgets.TextInput(name='Keeled', description='Eralda keeled komadega'),
     "keywords": pn.widgets.TextInput(name='Märksõnad', description='Eralda märksõnad komadega'),
     "is_valid": pn.widgets.Checkbox(name='Kehtetu', value=False)}
)

DEFAULT_METADATA = {"publication": "",
              "publication_year": 2024,
              "title": "",
              "author": "",
              "languages": [],
              "keywords": [],
              "is_valid": False}

class DataHandler(param.Parameterized):

    action = param.Action(
        default=lambda x: x.param.trigger('action'), label='Salvesta')

    user_input_widgets = collections.OrderedDict()

    # PDF browsing
    pdf_input = pn.widgets.FileInput(accept='.pdf', name='Vali fail', filename='', width=INPUT_WIDTH)
    pdf_viewer = pn.pane.PDF(pdf_input, width=500, height=800)

    submit_button = pn.widgets.Button(button_type='primary', name='Salvesta', align='start', width=BUTTON_WIDTH)
    new_entry_button = pn.widgets.Button(button_type='primary', name='Uus fail', align='end', width=BUTTON_WIDTH)

    static_text = pn.widgets.StaticText(name='Static Text', value='A string')
    alert_pane = pn.pane.Alert(object = 'Sisesta fail!', alert_type='warning', width = INPUT_WIDTH, height = 70, visible = False)

    metadata = copy.deepcopy(DEFAULT_METADATA)

    def __init__(self,**params):
        super().__init__(**params)
        self.user_input_widgets['filename'] = pn.widgets.TextInput(name='Faili pealkiri', placeholder=self.pdf_input.param.filename, disabled=True)
        self.user_input_widgets.update({meta_field: widget 
                                for meta_field, widget in USER_INPUT_FIELDS.items()})

    @param.depends('submit_button.value')
    def refresh_metadata(self):
        self.submit_button.disabled=True
        self.new_entry_button.disabled=True
        time.sleep(3)
        for name, widget in self.user_input_widgets.items():
            field_input = widget.value
            if name == 'is_valid':
                self.metadata[name] = not field_input
            elif name in ['keywords', 'languages']:
                self.metadata[name] = normalized_input_lists(widget.value)
            elif name == 'filename':
                self.metadata[name] = widget.placeholder
            elif isinstance(widget.value, str):
                self.metadata[name] = field_input.strip()
            else:
                self.metadata[name] = field_input

        if self.pdf_input.value:
            status = upload_to_db(self.pdf_input.value, self.metadata)
            if status == -1:
                self.alert_pane.param.update(object = 'Sellise pealkirjaga fail on juba baasis olemas!', visible=True)
        else:
            self.alert_pane.param.update(visible=True)
        
        self.new_entry_button.disabled=False
            
            
    @param.depends('new_entry_button.value')
    def new_input(self):
        """
        Resetting fields to upload a new file.
        """
        # Todo: uue saab alles siis sisestada, kui json on valmis ja metaandmed postgres? 
        self.alert_pane.param.update(visible=False)
        for name, widget in self.user_input_widgets.items():
            if isinstance(widget, pn.widgets.TextInput):
                widget.value = ""
            else:
                widget.value = DEFAULT_METADATA[name]
        if self.pdf_input.value:
            self.pdf_input.value = None
            self.pdf_input.filename = ''

        self.metadata = copy.deepcopy(DEFAULT_METADATA)
        self.submit_button.disabled=False

def file_upload():

    metadata = DataHandler()

    right_column = pn.Column(metadata.pdf_viewer)
    left_column = pn.Column(metadata.pdf_input,
                            *metadata.user_input_widgets.values(),
                            pn.Row(metadata.submit_button, metadata.refresh_metadata, 
                                    metadata.new_entry_button, metadata.new_input),
                            metadata.alert_pane
                            )

    page_layout = pn.Row(left_column, right_column)

    return page_layout