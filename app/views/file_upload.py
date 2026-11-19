import collections
import copy
import json

import httpx
import panel as pn

from app.telemetry import backend_client, ui_event, ui_log
from utils import observability as obs

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
        # Re-enable "Salvesta" whenever a new file arrives; previously it
        # stayed disabled after an upload until "Uus fail" was clicked.
        self.pdf_input.param.watch(self._on_file_selected, "value")

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

    def _on_file_selected(self, event) -> None:
        """A (new) file arrived from the browser: allow saving it."""
        if event.new is None:
            return
        ui_log("upload.file_selected",
               upload={"filename": self.pdf_input.filename, "size_bytes": len(event.new)})
        self.submit_button.disabled = False
        self.alert_pane.param.update(visible=False)

    def _collect_metadata(self) -> None:
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
                # Read straight from the file input: the placeholder binding
                # breaks after "Uus fail", which left the filename empty and
                # made the backend reject the upload (422/400).
                self.metadata[name] = self.pdf_input.filename or widget.placeholder or "document.pdf"
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

    def _show_result(self, ev, alert_type: str, message: str) -> None:
        """Show the outcome to the user and record exactly what they saw."""
        self.progress_bar.param.update(visible=False)
        self.status_text.param.update(visible=False)
        self.alert_pane.param.update(object=message, alert_type=alert_type, visible=True)
        ev.add(shown={"alert": alert_type, "message": obs.trunc(message, 300)})

    async def refresh_metadata(self, event) -> None:
        """Upload the selected PDF with its metadata via the backend.

        Async so the Panel server keeps serving other sessions while the
        backend processes the file (that can take minutes).
        """
        self.submit_button.disabled = True
        self.new_entry_button.disabled = True
        self.alert_pane.param.update(visible=False)
        self._collect_metadata()

        file_bytes = self.pdf_input.value
        with ui_event("upload.submit", upload={
            "filename": self.metadata.get("filename"),
            "size_bytes": len(file_bytes) if file_bytes else 0,
            "title": obs.trunc(self.metadata.get("title", ""), 200),
            "document_type": self.metadata.get("document_type"),
            "languages": ",".join(self.metadata.get("languages", [])),
            "year": self.metadata.get("publication_year"),
        }) as ev:
            try:
                if not file_bytes:
                    ev.set_outcome("no_file")
                    self._show_result(ev, "warning", "Sisesta fail!")
                    self.submit_button.disabled = False
                    return
                await self._upload(ev, file_bytes)
            finally:
                self.new_entry_button.disabled = False

    async def _upload(self, ev, file_bytes: bytes) -> None:
        self.progress_bar.param.update(visible=True, value=0)
        self.status_text.param.update(object="**Üleslaadimine algab...**", visible=True)

        valid_until_str = ""
        if self.metadata.get("valid_until"):
            valid_until_str = self.metadata["valid_until"].strftime("%Y-%m-%d")

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
        files = {"file": (self.metadata["filename"], file_bytes, "application/pdf")}

        final_result = None
        last_stage = None
        try:
            async with backend_client(timeout=600.0) as client:
                async with client.stream("POST", "/documents/upload-stream",
                                         data=form_data, files=files) as response:
                    ev.add(http_status=response.status_code)
                    if response.status_code != 200:
                        await response.aread()
                        try:
                            detail = response.json().get("detail", response.text)
                        except Exception:
                            detail = response.text
                        ev.set_outcome("rejected")
                        ev.add(rejection=obs.trunc(str(detail), 500))
                        self._show_result(ev, "warning",
                                          f"Server keeldus failist (HTTP {response.status_code}): {detail}")
                        return

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except ValueError as parse_err:
                            ev.incr("sse_parse_errors")
                            ev.add(sse_parse_error=obs.trunc(str(parse_err), 200))
                            continue
                        stage = data.get("stage", "")
                        if stage != "keepalive":
                            last_stage = stage
                        percentage = data.get("percentage", 0)
                        if percentage >= 0:
                            self.progress_bar.value = percentage
                        self.status_text.object = f"**{data.get('message', '')}**"
                        if stage in ("complete", "error"):
                            final_result = data
                            break
        except httpx.ConnectError as exc:
            ev.set_error(exc, reason="backend unreachable")
            self._show_result(ev, "warning", "Ei saa ühendust serveriga!")
            return
        except Exception as exc:
            ev.set_error(exc)
            self._show_result(ev, "warning", f"Viga: {str(exc)}")
            return
        finally:
            ev.add(last_stage=last_stage)

        if final_result is None:
            ev.set_outcome("unconfirmed")
            self._show_result(ev, "warning",
                              "Server ei kinnitanud üleslaadimist. Kontrolli allikate loendit.")
        elif final_result.get("status") == "error" or final_result.get("stage") == "error":
            message = final_result.get("message", "Üleslaadimine ebaõnnestus")
            ev.set_outcome("duplicate_filename" if "already exists" in message else "error")
            ev.add(backend_message=obs.trunc(message, 300))
            self._show_result(ev, "warning", message)
        else:
            ev.add(document_id=final_result.get("document_id"))
            msg = final_result.get("message", "Fail edukalt üles laetud!")
            self._show_result(ev, "success", f"Edukalt üles laetud: {msg}")

    def new_input(self, event) -> None:
        """ Resets the input fields to their default values for uploading a new file. """
        self.alert_pane.param.update(visible=False)
        
        for name, widget in self.user_input_widgets.items():
            if name == "filename":
                # Don't assign placeholder here: that unlinks it from
                # pdf_input.filename. Resetting pdf_input below clears it.
                pass
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
