import collections
import copy
import logging
import os

import httpx
import panel as pn

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

BUTTON_WIDTH = 125
INPUT_WIDTH = 300

# Document type options (same as document_management.py)
DOCUMENT_TYPE_OPTIONS = {
    "": "",
    "Õigusakt": "legal_act",
    "Õppematerjal": "educational",
    "Lõputöö": "thesis",
    "Erialaartikkel": "article",
    "Erialasõnastik": "glossary",
    "Meedia": "media",
    "Sotsiaalmeedia": "social_media",
    "Muu": "other",
}

DEFAULT_METADATA = {
    "document_type": "",
    "publication": "",
    "publication_year": 2024,
    "title": "",
    "short_name": "",
    "author": "",
    "languages": [],
    "keywords": [],
    "is_valid": True,
    "is_translation": False,
    "valid_until": None,
    "url": "",
}


class DataHandler:
    """
    A class that handles user input for uploading PDF files and associated metadata.

    Attributes:
        user_input_widgets (OrderedDict): A dictionary containing all of the user input widgets for metadata fields.
        pdf_input (pn.widgets.FileInput): A file input widget for selecting a PDF file.
        pdf_viewer (pn.pane.PDF): A panel for displaying the selected PDF file.
        submit_button (pn.widgets.Button): A button for submitting the user input and uploading the file.
        new_entry_button (pn.widgets.Button): A button for resetting the input fields to upload a new file.
        alert_pane (pn.pane.Alert): An alert pane for displaying warning messages.
        metadata (dict): A dictionary containing the metadata for the uploaded file.
    """

    def __init__(self, **params):
        self.metadata = copy.deepcopy(DEFAULT_METADATA)
        user_input_fields = collections.OrderedDict(
            {
                "title": pn.widgets.TextInput(name="Pealkiri *", width=INPUT_WIDTH),
                "short_name": pn.widgets.TextInput(
                    name="Lühinimi", 
                    placeholder="nt KVKS",
                    width=INPUT_WIDTH
                ),
                "document_type": pn.widgets.Select(
                    name="Dokumendi tüüp",
                    options=DOCUMENT_TYPE_OPTIONS,
                    width=INPUT_WIDTH,
                ),
                "author": pn.widgets.TextInput(name="Autor", width=INPUT_WIDTH),
                "publication": pn.widgets.TextInput(name="Väljaandja", width=INPUT_WIDTH),
                "publication_year": pn.widgets.IntInput(
                    name="Aasta", value=2024, start=1900, end=2100, width=100
                ),
                "url": pn.widgets.TextInput(name="Veebilink", width=INPUT_WIDTH),
                "languages": pn.widgets.TextInput(
                    name="Keeled", 
                    placeholder="et, en, de",
                    width=INPUT_WIDTH
                ),
                "is_translation": pn.widgets.Checkbox(name="Dokument on tõlge", value=False),
                "keywords": pn.widgets.TextInput(
                    name="Märksõnad", 
                    placeholder="märksõna1, märksõna2",
                    width=INPUT_WIDTH
                ),
                "is_valid": pn.widgets.Checkbox(name="Kehtetu", value=False),
                "valid_until": pn.widgets.DatePicker(
                    name="Kehtiv kuni",
                    width=180,
                ),
            }
        )

        self.user_input_widgets = collections.OrderedDict()
        self.pdf_input = pn.widgets.FileInput(
            accept=".pdf", name="Vali fail", filename="", width=INPUT_WIDTH
        )
        self.pdf_viewer = pn.pane.PDF(self.pdf_input, width=500, height=800)

        self.user_input_widgets["filename"] = pn.widgets.TextInput(
            name="Faili pealkiri",
            placeholder=self.pdf_input.param.filename,
            disabled=True,
        )
        self.user_input_widgets.update(user_input_fields)

        # Buttons and button actions
        self.submit_button = pn.widgets.Button(
            button_type="primary", name="Salvesta", align="start", width=BUTTON_WIDTH
        )
        self.new_entry_button = pn.widgets.Button(
            button_type="primary", name="Uus fail", align="end", width=BUTTON_WIDTH
        )
        self.submit_button.on_click(self.refresh_metadata)
        self.new_entry_button.on_click(self.new_input)

        self.alert_pane = pn.pane.Alert(
            object="Sisesta fail!",
            alert_type="warning",
            width=INPUT_WIDTH,
            height=70,
            visible=False,
        )
        
        # Progress indicator for upload
        self.progress_bar = pn.indicators.Progress(
            name="Üleslaadimine...",
            active=True,
            bar_color="primary",
            width=INPUT_WIDTH,
            visible=False,
        )
        self.status_text = pn.pane.Markdown(
            "",
            width=INPUT_WIDTH,
            visible=False,
        )

        right_column = pn.Column(self.pdf_viewer)
        left_column = pn.Column(
            self.pdf_input,
            *self.user_input_widgets.values(),
            pn.Row(self.submit_button, self.new_entry_button),
            self.progress_bar,
            self.status_text,
            self.alert_pane,
        )

        self.page_layout = pn.Row(left_column, right_column)
        super().__init__(**params)

    def refresh_metadata(self, event) -> None:
        """ Updates the metadata dictionary with the user input values and uploads via backend API. """
        self.submit_button.disabled = True
        self.new_entry_button.disabled = True
        self.alert_pane.param.update(visible=False)
        logger.info("Saving triggered")
        
        for name, widget in self.user_input_widgets.items():
            field_input = widget.value
            if name == "is_valid":
                # Checkbox is "Kehtetu" (invalid), so invert it
                self.metadata[name] = not field_input
            elif name == "keywords":
                # Parse comma-separated keywords
                kw_input = widget.value_input if hasattr(widget, 'value_input') else widget.value
                if kw_input:
                    self.metadata[name] = [k.strip() for k in kw_input.split(",") if k.strip()]
                else:
                    self.metadata[name] = []
            elif name == "languages":
                # Parse comma-separated languages
                if widget.value:
                    self.metadata[name] = [l.strip() for l in widget.value.split(",") if l.strip()]
                else:
                    self.metadata[name] = []
            elif name == "filename":
                self.metadata[name] = widget.placeholder
            elif name == "valid_until":
                # DatePicker returns date object or None
                self.metadata[name] = field_input
            elif name == "document_type":
                # Select returns the value (API key), not the label
                self.metadata[name] = field_input if field_input else ""
            elif isinstance(widget.value, str):
                self.metadata[name] = field_input.strip() if field_input else ""
            else:
                self.metadata[name] = field_input

        if self.pdf_input.value:
            # Show progress indicator
            self.progress_bar.param.update(visible=True, value=0)
            self.status_text.param.update(
                object="**Üleslaadimine algab...**",
                visible=True,
            )
            
            try:
                # Prepare valid_until for API (as string or empty)
                valid_until_str = ""
                if self.metadata.get("valid_until"):
                    valid_until_str = self.metadata["valid_until"].strftime("%Y-%m-%d")
                
                # Use streaming endpoint with SSE for progress
                form_data = {
                    "title": self.metadata.get("title", ""),
                    "short_name": self.metadata.get("short_name", ""),
                    "document_type": self.metadata.get("document_type", ""),
                    "author": self.metadata.get("author", ""),
                    "publication": self.metadata.get("publication", ""),
                    "publication_year": str(self.metadata.get("publication_year", 2024)),
                    "url": self.metadata.get("url", ""),
                    "languages": ",".join(self.metadata.get("languages", [])),
                    "is_translation": str(self.metadata.get("is_translation", False)).lower(),
                    "keywords": ",".join(self.metadata.get("keywords", [])),
                    "is_valid": str(self.metadata.get("is_valid", True)).lower(),
                    "valid_until": valid_until_str,
                }
                
                files = {"file": (self.metadata["filename"], self.pdf_input.value, "application/pdf")}
                
                # Stream the response to get progress updates
                with httpx.Client(timeout=600.0) as client:
                    with client.stream(
                        "POST",
                        f"{BACKEND_URL}/documents/upload-stream",
                        data=form_data,
                        files=files,
                    ) as response:
                        final_result = None
                        for line in response.iter_lines():
                            if line.startswith("data: "):
                                try:
                                    import json
                                    data = json.loads(line[6:])
                                    stage = data.get("stage", "")
                                    percentage = data.get("percentage", 0)
                                    message = data.get("message", "")
                                    
                                    # Update progress bar
                                    if percentage >= 0:
                                        self.progress_bar.value = percentage
                                    self.status_text.object = f"**{message}**"
                                    
                                    # Check for completion
                                    if stage in ("complete", "error"):
                                        final_result = data
                                        break
                                except Exception as parse_err:
                                    logger.warning(f"Parse error: {parse_err}")
                        
                        # Handle final result
                        self.progress_bar.param.update(visible=False)
                        self.status_text.param.update(visible=False)
                        
                        if final_result:
                            if final_result.get("status") == "error" or final_result.get("stage") == "error":
                                self.alert_pane.param.update(
                                    object=final_result.get("message", "Üleslaadimine ebaõnnestus"),
                                    alert_type="warning",
                                    visible=True,
                                )
                            else:
                                msg = final_result.get("message", "Fail edukalt üles laetud!")
                                self.alert_pane.param.update(
                                    object=f"Edukalt üles laetud: {msg}",
                                    alert_type="success",
                                    visible=True,
                                )
                        else:
                            self.alert_pane.param.update(
                                object="Üleslaadimine lõpetatud",
                                alert_type="success",
                                visible=True,
                            )
                            
            except httpx.ConnectError:
                logger.error("Cannot connect to backend")
                self.progress_bar.param.update(visible=False)
                self.status_text.param.update(visible=False)
                self.alert_pane.param.update(
                    object="Ei saa ühendust serveriga!",
                    alert_type="warning",
                    visible=True,
                )
            except Exception as e:
                logger.error(f"Upload error: {e}")
                self.progress_bar.param.update(visible=False)
                self.status_text.param.update(visible=False)
                self.alert_pane.param.update(
                    object=f"Viga: {str(e)}",
                    alert_type="warning",
                    visible=True,
                )
        else:
            self.alert_pane.param.update(visible=True)

        self.new_entry_button.disabled = False

    def new_input(self, event) -> None:
        """ Resets the input fields to their default values for uploading a new file. """
        logger.info("New input triggered")
        self.alert_pane.param.update(visible=False)
        
        for name, widget in self.user_input_widgets.items():
            if name == "filename":
                widget.placeholder = ""
            elif isinstance(widget, pn.widgets.TextInput):
                widget.value = ""
                if hasattr(widget, 'value_input'):
                    widget.value_input = ""
            elif isinstance(widget, pn.widgets.Select):
                widget.value = ""
            elif isinstance(widget, pn.widgets.Checkbox):
                widget.value = DEFAULT_METADATA.get(name, False)
            elif isinstance(widget, pn.widgets.DatePicker):
                widget.value = None
            elif isinstance(widget, pn.widgets.IntInput):
                widget.value = DEFAULT_METADATA.get(name, 2024)
            else:
                default = DEFAULT_METADATA.get(name)
                if default is not None:
                    widget.value = default
                    
        if self.pdf_input.param.value:
            self.pdf_input.param.update(value=None, filename="")

        self.metadata = copy.deepcopy(DEFAULT_METADATA)
        self.submit_button.disabled = False


def file_upload() -> pn.layout.ListPanel:
    """
    Creates and returns the layout of the user interface for file upload page.

    Returns:
        pn.Row: A row layout containing the left column (file input, metadata input fields, buttons, and alert pane)
                and the right column (PDF viewer).
    """

    metadata = DataHandler()
    return metadata.page_layout
