"""
Document Management View - List, edit, and delete documents with Qdrant sync.
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import httpx
import panel as pn
import param

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Layout constants
FORM_WIDTH = 300
BUTTON_WIDTH = 100

pn.extension("tabulator")

# Document type labels (Estonian)
DOCUMENT_TYPE_LABELS = {
    "legal_act": "Õigusakt",
    "educational": "Õppematerjal",
    "thesis": "Lõputöö",
    "article": "Erialaartikkel",
    "glossary": "Erialasõnastik",
    "media": "Meedia",
    "social_media": "Sotsiaalmeedia",
    "other": "Muu",
}

DOCUMENT_TYPE_OPTIONS = {v: k for k, v in DOCUMENT_TYPE_LABELS.items()}


class DocumentManagementView(param.Parameterized):
    """Document management interface with list, edit, and delete functionality."""
    
    refresh_trigger = param.Integer(default=0)
    
    def __init__(self, **params):
        super().__init__(**params)
        self.selected_doc_id = None
        self.documents = []
        self._create_widgets()
        self._create_layout()
        # Load documents on init
        try:
            if pn.state.curdoc and pn.state.curdoc.session_context:
                pn.state.execute(self._load_documents_sync)
            else:
                self._load_documents_sync()
        except Exception as e:
            logger.warning(f"Could not load documents on init: {e}")

    def _create_widgets(self):
        """Create UI widgets."""
        
        # Document table
        self.documents_table = pn.widgets.Tabulator(
            value=None,
            sizing_mode="stretch_width",
            height=400,
            show_index=False,
            selectable=1,
            pagination="local",
            page_size=15,
            theme="simple",
            configuration={
                "columns": [
                    {"field": "id", "title": "ID", "width": 50},
                    {"field": "title", "title": "Pealkiri", "width": 300},
                    {"field": "short_name", "title": "Lühinimi", "width": 80},
                    {"field": "document_type_label", "title": "Tüüp", "width": 100},
                    {"field": "author", "title": "Autor", "width": 120},
                    {"field": "publication_year", "title": "Aasta", "width": 60},
                    {"field": "validity_label", "title": "Kehtivus", "width": 70},
                    {"field": "chunk_count", "title": "Lõike", "width": 55},
                ],
            },
        )
        self.documents_table.on_click(self._on_row_select)
        
        # Buttons
        self.refresh_btn = pn.widgets.Button(
            name="Värskenda", button_type="light", width=BUTTON_WIDTH
        )
        self.refresh_btn.on_click(self._on_refresh)
        
        self.edit_btn = pn.widgets.Button(
            name="Muuda", button_type="primary", width=BUTTON_WIDTH, disabled=True
        )
        self.edit_btn.on_click(self._on_edit)
        
        self.delete_btn = pn.widgets.Button(
            name="Kustuta", button_type="danger", width=BUTTON_WIDTH, disabled=True
        )
        self.delete_btn.on_click(self._on_delete)
        
        # Edit form
        self._create_edit_form()
        
        # Feedback
        self.alert = pn.pane.Alert("", alert_type="info", visible=False)
        self.loading = pn.indicators.LoadingSpinner(value=False, width=25, height=25)

    def _create_edit_form(self):
        """Create the edit form widgets."""
        self.edit_title = pn.widgets.TextInput(name="Pealkiri *", width=FORM_WIDTH)
        self.edit_short_name = pn.widgets.TextInput(name="Lühinimi", width=FORM_WIDTH, placeholder="nt KVKS")
        self.edit_document_type = pn.widgets.Select(
            name="Dokumendi tüüp", options=DOCUMENT_TYPE_OPTIONS, width=FORM_WIDTH
        )
        self.edit_author = pn.widgets.TextInput(name="Autor", width=FORM_WIDTH)
        self.edit_publication = pn.widgets.TextInput(name="Väljaandja", width=FORM_WIDTH)
        self.edit_publication_year = pn.widgets.IntInput(
            name="Aasta", value=2024, start=1900, end=2100, width=100
        )
        self.edit_url = pn.widgets.TextInput(name="Veebilink", width=FORM_WIDTH)
        self.edit_languages = pn.widgets.TextInput(
            name="Keeled", placeholder="et, en, de", width=FORM_WIDTH
        )
        self.edit_is_translation = pn.widgets.Checkbox(name="Dokument on tõlge", value=False)
        self.edit_keywords = pn.widgets.TextInput(
            name="Märksõnad", placeholder="märksõna1, märksõna2", width=FORM_WIDTH
        )
        self.edit_is_valid = pn.widgets.Checkbox(name="Kehtetu", value=False)
        self.edit_valid_until = pn.widgets.DatePicker(name="Kehtiv kuni", width=FORM_WIDTH)
        
        self.save_btn = pn.widgets.Button(
            name="Salvesta", button_type="success", width=BUTTON_WIDTH
        )
        self.save_btn.on_click(self._on_save)
        
        self.cancel_btn = pn.widgets.Button(
            name="Tühista", button_type="light", width=BUTTON_WIDTH
        )
        self.cancel_btn.on_click(self._on_cancel)
        
        self.edit_form = pn.Column(
            self.edit_title,
            self.edit_short_name,
            self.edit_document_type,
            self.edit_author,
            self.edit_publication,
            self.edit_publication_year,
            self.edit_url,
            self.edit_languages,
            self.edit_is_translation,
            self.edit_keywords,
            self.edit_is_valid,
            self.edit_valid_until,
            pn.Row(self.save_btn, self.cancel_btn),
            visible=False,
            width=FORM_WIDTH + 20,
        )

    def _create_layout(self):
        """Create the complete layout."""
        
        # Button row
        button_row = pn.Row(
            self.loading,
            self.refresh_btn,
            self.edit_btn,
            self.delete_btn,
            align="center",
        )
        
        # Main content
        self.layout = pn.Column(
            button_row,
            self.alert,
            pn.Row(
                self.documents_table,
                pn.Spacer(width=20),
                self.edit_form,
            ),
            sizing_mode="stretch_width",
            margin=10,
        )

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _load_documents_sync(self):
        """Load documents from backend (synchronous)."""
        self.loading.value = True
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(f"{BACKEND_URL}/documents")
                response.raise_for_status()
                self._process_documents_response(response.json())
        except httpx.ConnectError:
            logger.warning("Cannot connect to backend")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
        finally:
            self.loading.value = False

    async def _load_documents(self):
        """Load documents from backend (async)."""
        self.loading.value = True
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{BACKEND_URL}/documents")
                response.raise_for_status()
                self._process_documents_response(response.json())
                self._show_alert(f"{len(self.documents)} dokumenti", "success", auto_hide=True)
        except httpx.ConnectError:
            self._show_alert("Ei saa ühendust serveriga", "danger")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            self._show_alert(f"Viga: {str(e)}", "danger")
        finally:
            self.loading.value = False

    def _process_documents_response(self, data: dict):
        """Process documents response and update table."""
        import pandas as pd
        
        self.documents = data.get("documents", [])
        table_data = []
        for doc in self.documents:
            doc_type = doc.get("document_type", "other")
            is_valid = doc.get("is_valid", True)
            valid_until = doc.get("valid_until")
            
            # Check if expired
            if valid_until:
                try:
                    valid_date = datetime.strptime(valid_until[:10], "%Y-%m-%d").date()
                    if valid_date < datetime.now().date():
                        is_valid = False
                except ValueError:
                    pass
            
            table_data.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "short_name": doc.get("short_name", "") or "",
                "document_type": doc_type,
                "document_type_label": DOCUMENT_TYPE_LABELS.get(doc_type, doc_type),
                "author": doc.get("author", "") or "",
                "publication_year": doc.get("publication_year"),
                "is_valid": is_valid,
                "validity_label": "Kehtiv" if is_valid else "Kehtetu",
                "valid_until": valid_until,
                "chunk_count": doc.get("chunk_count", 0),
            })
        
        self.documents_table.value = pd.DataFrame(table_data)

    async def _load_document_details(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Load single document details."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BACKEND_URL}/documents/{doc_id}")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_row_select(self, event):
        """Handle row selection."""
        if event.row is not None:
            row_data = self.documents_table.value.iloc[event.row]
            self.selected_doc_id = row_data["id"]
            self.edit_btn.disabled = False
            self.delete_btn.disabled = False
        else:
            self.selected_doc_id = None
            self.edit_btn.disabled = True
            self.delete_btn.disabled = True

    def _on_refresh(self, event):
        """Refresh document list."""
        asyncio.create_task(self._load_documents())

    def _on_edit(self, event):
        """Open edit form."""
        if self.selected_doc_id:
            asyncio.create_task(self._open_edit_form())

    async def _open_edit_form(self):
        """Load document and populate edit form."""
        doc = await self._load_document_details(self.selected_doc_id)
        if not doc:
            self._show_alert("Dokumendi laadimine ebaõnnestus", "danger")
            return
        
        # Populate form
        self.edit_title.value = doc.get("title", "")
        self.edit_short_name.value = doc.get("short_name", "") or ""
        self.edit_document_type.value = doc.get("document_type", "other")
        self.edit_author.value = doc.get("author", "") or ""
        self.edit_publication.value = doc.get("publication", "") or ""
        self.edit_publication_year.value = doc.get("publication_year", 2024) or 2024
        self.edit_url.value = doc.get("url", "") or ""
        self.edit_languages.value = doc.get("languages", "") or ""
        self.edit_is_translation.value = doc.get("is_translation", False)
        
        keywords = doc.get("keywords", [])
        self.edit_keywords.value = ", ".join(keywords) if keywords else ""
        
        self.edit_is_valid.value = not doc.get("is_valid", True)  # Checkbox is "Kehtetu"
        
        valid_until = doc.get("valid_until")
        if valid_until:
            try:
                self.edit_valid_until.value = datetime.strptime(valid_until[:10], "%Y-%m-%d").date()
            except ValueError:
                self.edit_valid_until.value = None
        else:
            self.edit_valid_until.value = None
        
        self.edit_form.visible = True

    def _on_cancel(self, event):
        """Cancel editing."""
        self.edit_form.visible = False

    def _on_save(self, event):
        """Save document changes."""
        asyncio.create_task(self._save_document())

    async def _save_document(self):
        """Send document updates to backend."""
        if not self.selected_doc_id:
            return
        
        self.loading.value = True
        self._hide_alert()
        
        try:
            keywords = []
            if self.edit_keywords.value:
                keywords = [k.strip() for k in self.edit_keywords.value.split(",") if k.strip()]
            
            valid_until = None
            if self.edit_valid_until.value:
                valid_until = self.edit_valid_until.value.strftime("%Y-%m-%d")
            
            payload = {
                "title": self.edit_title.value,
                "short_name": self.edit_short_name.value or None,
                "document_type": self.edit_document_type.value,
                "author": self.edit_author.value or None,
                "publication": self.edit_publication.value or None,
                "publication_year": self.edit_publication_year.value,
                "url": self.edit_url.value or None,
                "languages": self.edit_languages.value or None,
                "is_translation": self.edit_is_translation.value,
                "keywords": keywords if keywords else None,
                "is_valid": not self.edit_is_valid.value,  # Invert: checkbox is "Kehtetu"
                "valid_until": valid_until,
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.put(
                    f"{BACKEND_URL}/documents/{self.selected_doc_id}",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    self._show_alert("Salvestatud", "success", auto_hide=True)
                    self.edit_form.visible = False
                    await self._load_documents()
                else:
                    self._show_alert(result.get("message", "Viga"), "danger")
                    
        except httpx.ConnectError:
            self._show_alert("Ei saa ühendust serveriga", "danger")
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            self._show_alert(f"Viga: {str(e)}", "danger")
        finally:
            self.loading.value = False

    def _on_delete(self, event):
        """Show delete confirmation."""
        if not self.selected_doc_id:
            return
        
        title = "dokument"
        for doc in self.documents:
            if doc.get("id") == self.selected_doc_id:
                title = doc.get("title", "dokument")
                break
        
        confirm_text = pn.pane.Markdown(
            f"Kustutada **'{title}'**?\n\nKustutab ka lõigud Qdrant'ist.",
            width=350,
        )
        
        confirm_btn = pn.widgets.Button(name="Kustuta", button_type="danger", width=90)
        cancel_btn = pn.widgets.Button(name="Tühista", button_type="light", width=90)
        
        dialog = pn.Column(
            confirm_text,
            pn.Row(confirm_btn, cancel_btn),
            styles={"background": "#fff3cd", "padding": "10px", "border-radius": "5px"},
        )
        
        self.layout.insert(2, dialog)
        
        def on_confirm(e):
            asyncio.create_task(self._delete_document())
            self.layout.remove(dialog)
        
        def on_cancel(e):
            self.layout.remove(dialog)
        
        confirm_btn.on_click(on_confirm)
        cancel_btn.on_click(on_cancel)

    async def _delete_document(self):
        """Delete document."""
        if not self.selected_doc_id:
            return
        
        self.loading.value = True
        self._hide_alert()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.delete(f"{BACKEND_URL}/documents/{self.selected_doc_id}")
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    self._show_alert("Kustutatud", "success", auto_hide=True)
                    self.selected_doc_id = None
                    self.edit_btn.disabled = True
                    self.delete_btn.disabled = True
                    await self._load_documents()
                else:
                    self._show_alert(result.get("message", "Viga"), "danger")
                    
        except httpx.ConnectError:
            self._show_alert("Ei saa ühendust serveriga", "danger")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            self._show_alert(f"Viga: {str(e)}", "danger")
        finally:
            self.loading.value = False

    # =========================================================================
    # Helpers
    # =========================================================================

    def _show_alert(self, message: str, alert_type: str = "info", auto_hide: bool = False):
        """Show alert message."""
        self.alert.object = message
        self.alert.alert_type = alert_type
        self.alert.visible = True
        
        if auto_hide:
            async def hide():
                await asyncio.sleep(2)
                self._hide_alert()
            asyncio.create_task(hide())

    def _hide_alert(self):
        """Hide alert."""
        self.alert.visible = False


def document_management() -> pn.Column:
    """Create and return the document management view."""
    view = DocumentManagementView()
    return view.layout
