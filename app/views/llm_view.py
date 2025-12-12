"""
LLM View - Panel frontend that calls the FastAPI backend.
No heavy model loading here - all processing done by backend.
"""
import asyncio
import logging
import os
from typing import Optional

import httpx
import panel as pn
import param
from panel.chat import ChatInterface, ChatMessage
import pandas as pd

from utils.db_connection import Connection

# Custom CSS for See Also buttons - styled as pill badges
SEE_ALSO_CSS = """
.see-also-btn button.bk-btn,
.see-also-btn .bk-btn {
    background: #e3f2fd !important;
    color: #1565c0 !important;
    border: 1px solid #90caf9 !important;
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 0.9em !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    box-shadow: none !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.see-also-btn button.bk-btn:hover,
.see-also-btn .bk-btn:hover {
    background: #1565c0 !important;
    color: white !important;
    transform: scale(1.05);
    border-color: #1565c0 !important;
}
.see-also-btn button.bk-btn:active,
.see-also-btn .bk-btn:active {
    transform: scale(0.98);
}
.see-also-label {
    color: #666;
    font-weight: 600;
    margin-right: 8px;
    font-size: 0.9em;
}
"""

# Initialize Panel extension with perspective and our custom CSS
pn.extension("perspective", raw_css=[SEE_ALSO_CSS])

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

CONNECTION_PARAMS = {
    "host": os.getenv("PG_HOST"),
    "port": os.getenv("PG_PORT"),
    "user": os.getenv("PG_USER"),
    "password": os.getenv("PG_PASSWORD"),
    "db": os.getenv("PG_COLLECTION"),
}

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)


class BackendClient:
    """
    Client for communicating with the FastAPI backend.
    """
    
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        self.timeout = 120.0  # 2 minutes timeout for LLM calls
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        files: list = None,
        only_valid: bool = False,
    ) -> dict:
        """Call the search endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "filters": {
                        "limit": limit,
                        "files": files or [],
                        "only_valid": only_valid,
                    }
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def chat(
        self,
        query: str,
        limit: int = 5,
        files: list = None,
        only_valid: bool = False,
        debug: bool = False,
        parallel: bool = False,
        expand_query: bool = False,
        expand_context: bool = False,
        use_reranking: bool = True,
        output_categories: list = None,
    ) -> dict:
        """Call the chat endpoint."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat",
                json={
                    "query": query,
                    "filters": {
                        "limit": limit,
                        "files": files or [],
                        "only_valid": only_valid,
                    },
                    "debug": debug,
                    "parallel": parallel,
                    "expand_query": expand_query,
                    "expand_context": expand_context,
                    "use_reranking": use_reranking,
                    "output_categories": output_categories or ["definitions", "related_terms", "usage_evidence"],
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_prompt_sets(self) -> list:
        """Get all available prompt sets."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/prompt-sets")
                response.raise_for_status()
                return response.json().get("prompt_sets", [])
        except Exception as e:
            logger.error(f"Failed to fetch prompt sets: {e}")
            return []
    
    async def get_prompt_set(self, set_id: int) -> dict:
        """Get a specific prompt set with its prompts."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/prompt-sets/{set_id}")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch prompt set: {e}")
            return {}
    
    async def get_prompts(self, set_id: int = None) -> list:
        """Get all available prompts, optionally filtered by set."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                url = f"{self.base_url}/prompts"
                if set_id is not None:
                    url += f"?set_id={set_id}"
                response = await client.get(url)
                response.raise_for_status()
                return response.json().get("prompts", [])
        except Exception as e:
            logger.error(f"Failed to fetch prompts: {e}")
            return []
    
    async def get_prompt(self, set_id: int, prompt_type: str) -> dict:
        """Get a specific prompt within a set."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/prompt-sets/{set_id}/prompts/{prompt_type}")
            response.raise_for_status()
            return response.json()
    
    async def update_prompt(self, set_id: int, prompt_type: str, prompt_text: str, description: str = None) -> dict:
        """Update a prompt within a set."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.put(
                f"{self.base_url}/prompt-sets/{set_id}/prompts/{prompt_type}",
                json={"prompt_text": prompt_text, "description": description}
            )
            response.raise_for_status()
            return response.json()
    
    async def create_prompt(self, set_id: int, prompt_type: str, prompt_text: str, description: str = None) -> dict:
        """Create a new prompt within a set."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/prompt-sets/{set_id}/prompts",
                json={"prompt_type": prompt_type, "prompt_text": prompt_text, "description": description}
            )
            response.raise_for_status()
            return response.json()


class FilterState:
    """
    Manages filter state for the chat.
    """
    def __init__(self):
        self.files: list = []
        self.limit: int = 8  # Increased from 5 for better term diversity
        self.only_valid: bool = False
        self.prompt_set_id: int = None  # Uses default prompt set if None
        self.debug_mode: bool = False
        self.parallel_mode: bool = True  # Enabled by default
        self.expand_query: bool = True   # Enabled by default
        self.expand_context: bool = True  # Enabled by default
        self.use_reranking: bool = True  # Enabled by default
        self.output_categories: list = ["definitions", "related_terms", "usage_evidence"]  # All enabled by default
    
    def apply(self, files: list, limit: int, only_valid: bool, prompt_set_id: int = None, debug_mode: bool = False, parallel_mode: bool = False, expand_query: bool = False, expand_context: bool = False, use_reranking: bool = True, output_categories: list = None):
        self.files = files
        self.limit = limit
        self.only_valid = only_valid
        self.prompt_set_id = prompt_set_id
        self.debug_mode = debug_mode
        self.parallel_mode = parallel_mode
        self.expand_query = expand_query
        self.expand_context = expand_context
        self.use_reranking = use_reranking
        self.output_categories = output_categories or ["definitions", "related_terms", "usage_evidence"]


class FilterActionHandler(param.Parameterized):
    """
    Handles filter UI and state management.
    """

    def __init__(self, filter_state: FilterState, backend_client: 'BackendClient', **params):
        self.filter_state = filter_state
        self.backend_client = backend_client

        self.con = Connection(**CONNECTION_PARAMS)
        self.con.establish_connection()

        self.apply_filters_button = pn.widgets.Button(
            name="Rakenda filtrid",
            button_type="primary",
            width=50,
            margin=(20, 60, 0, 0),
        )

        self.refresh_choices_button = pn.widgets.Button(
            name="Värskenda filtrid",
            button_type="primary",
            width=50,
            margin=(20, 0, 0, 20),
        )

        self.keyword_selector = pn.widgets.CrossSelector(
            name="Märksõnad", value=[], options=[], size=8, width=500
        )
        self.file_selector = pn.widgets.CrossSelector(
            name="Dokumendid", value=[], options=[], size=8, width=500
        )

        self.limit_slider = pn.widgets.EditableIntSlider(
            name="Tekstilõikude arv SKMi sisendis",
            start=1,
            end=20,
            step=1,
            value=8,  # Increased for better term diversity
            width=500,
        )

        self.validity_checkbox = pn.widgets.Checkbox(
            name="Otsi ainult kehtivatest", width=500
        )
        
        self.prompt_set_selector = pn.widgets.Select(
            name="Promptide komplekt",
            options={"Vaikimisi terminoloogiaanalüüs": None},  # None means default
            value=None,
            width=500,
        )
        
        self.debug_checkbox = pn.widgets.Checkbox(
            name="🔍 Debug mode (näita pipeline'i samme)", 
            width=500,
            value=False,
        )
        
        self.parallel_checkbox = pn.widgets.Checkbox(
            name="⚡ Parallel mode (3 spetsialiseeritud prompti paralleelselt)", 
            width=500,
            value=True,  # Enabled by default
        )
        
        self.expand_query_checkbox = pn.widgets.Checkbox(
            name="🔄 Query expansion (laienda päringut sünonüümide/seotud terminitega)", 
            width=500,
            value=True,  # Enabled by default
        )
        
        self.expand_context_checkbox = pn.widgets.Checkbox(
            name="📄 Context expansion (lisa külgnevad lõigud täielikuma konteksti jaoks)", 
            width=500,
            value=True,  # Enabled by default
        )
        
        self.reranking_checkbox = pn.widgets.Checkbox(
            name="🎯 Reranking (järjesta tulemused cross-encoder mudeliga täpsemaks)", 
            width=500,
            value=True,  # Enabled by default
        )
        
        self.output_categories_selector = pn.widgets.CheckBoxGroup(
            name="Väljundi kategooriad",
            value=["definitions", "related_terms", "usage_evidence"],  # All selected by default
            options={
                "📖 Definitsioonid": "definitions",
                "🔗 Seotud terminid": "related_terms",
                "📝 Kasutuskontekstid": "usage_evidence",
            },
            inline=False,
        )

        super().__init__(**params)
        self.refresh_selectors()
        # Load prompt sets asynchronously
        asyncio.create_task(self._load_prompt_sets())

    async def _load_prompt_sets(self):
        """Load prompt sets from backend."""
        try:
            prompt_sets = await self.backend_client.get_prompt_sets()
            if prompt_sets:
                options = {}
                default_set_id = None
                for ps in prompt_sets:
                    label = ps.get("name", "Unknown")
                    if ps.get("is_default"):
                        label += " ✓"
                        default_set_id = ps.get("id")
                    prompt_count = ps.get("prompt_count", 0)
                    label += f" ({prompt_count} prompti)"
                    options[label] = ps.get("id")
                self.prompt_set_selector.options = options
                # Set to default prompt set
                if default_set_id is not None:
                    self.prompt_set_selector.value = default_set_id
                logger.info(f"Loaded {len(prompt_sets)} prompt sets")
        except Exception as e:
            logger.error(f"Failed to load prompt sets: {e}")
    
    def load_keywords_from_db(self):
        try:
            keywords_df = self.con.statement_to_df(
                """ SELECT DISTINCT keyword FROM keywords ORDER BY keyword"""
            )
            return keywords_df
        except Exception as e:
            logger.error(e)
            return pd.DataFrame(columns=["keyword"])

    def load_files_from_db(self):
        try:
            files_df = self.con.statement_to_df(
                """ SELECT id, title FROM documents WHERE current_state = 'uploaded' ORDER BY title"""
            )
            return files_df
        except Exception as e:
            logger.error(e)
            return pd.DataFrame(columns=["id", "title"])

    @param.depends("refresh_choices_button.value", watch=True)
    def refresh_selectors(self):
        logger.info("Refreshing file selection")
        try:
            self.files_df = self.load_files_from_db()
            self.file_selector.options = list(self.files_df["title"])
            self.keywords_df = self.load_keywords_from_db()
            self.keyword_selector.options = list(self.keywords_df["keyword"])
            logger.info("File selection refresh complete.")
        except Exception as e:
            logger.error(e)

    @param.depends("keyword_selector.value", watch=True)
    def keyword_filtering(self):
        selected_kw_values = self.keyword_selector.value
        if selected_kw_values:
            logger.info(selected_kw_values)
            self.con.establish_connection()
            try:
                result = self.con.execute_sql(
                    """SELECT document_id FROM keywords WHERE keyword IN :kws""",
                    [{"kws": (tuple(selected_kw_values))}],
                )
                compatible_document_ids = [row[0] for row in result["data"]]
                selected_files = list(
                    self.files_df[self.files_df["id"].isin(compatible_document_ids)]["title"]
                )
                self.file_selector.options = selected_files
                self.file_selector.value = selected_files
            except Exception as e:
                logger.error(e)
        else:
            self.refresh_selectors()

    @param.depends("apply_filters_button.value", watch=True)
    def apply_filters(self):
        self.filter_state.apply(
            files=self.file_selector.value,
            limit=self.limit_slider.value,
            only_valid=self.validity_checkbox.value,
            prompt_set_id=self.prompt_set_selector.value,
            debug_mode=self.debug_checkbox.value,
            parallel_mode=self.parallel_checkbox.value,
            expand_query=self.expand_query_checkbox.value,
            expand_context=self.expand_context_checkbox.value,
            use_reranking=self.reranking_checkbox.value,
            output_categories=self.output_categories_selector.value,
        )
        logger.info(f"Filters applied: {self.filter_state.__dict__}")


def _format_debug_output(debug_info: dict) -> str:
    """Format debug information into a readable markdown string."""
    if not debug_info:
        return ""
    
    extraction_mode = debug_info.get('extraction_mode', 'single')
    mode_icon = "⚡" if extraction_mode == "parallel" else "📝"
    
    lines = []
    lines.append("---")
    lines.append(f"## 🔍 Debug: Pipeline Steps ({mode_icon} {extraction_mode} mode)")
    lines.append("")
    lines.append(f"**Total Duration:** {debug_info.get('total_duration_ms', 0):.1f} ms")
    lines.append(f"**Extraction Mode:** {extraction_mode}")
    lines.append(f"**Search Type:** {debug_info.get('search_type', 'unknown')}")
    lines.append(f"**LLM Model:** {debug_info.get('model_used', 'unknown')}")
    lines.append(f"**Embedding Model:** {debug_info.get('embedding_model', 'unknown')}")
    lines.append("")
    
    for step in debug_info.get("pipeline_steps", []):
        step_num = step.get("step_number", 0)
        step_name = step.get("step_name", "Unknown")
        description = step.get("description", "")
        duration = step.get("duration_ms", 0)
        data = step.get("data", "")
        
        lines.append(f"### Step {step_num}: {step_name}")
        lines.append(f"*{description}* ({duration:.1f} ms)")
        lines.append("")
        
        if data:
            lines.append("<details>")
            lines.append(f"<summary>Show data</summary>")
            lines.append("")
            lines.append("```")
            lines.append(data)
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    
    lines.append("---")
    return "\n".join(lines)


def llm_view():
    """
    Create the LLM chat view.
    This is now a lightweight frontend that calls the backend API.
    """
    # Initialize backend client and filter state
    backend_client = BackendClient()
    filter_state = FilterState()
    filter_handler = FilterActionHandler(filter_state, backend_client)

    # Helper to create See Also button click handler
    def create_see_also_handler(term: str, chat_interface):
        """Create a click handler for a See Also button."""
        def handler(event):
            logger.info(f"See Also button clicked: {term}")
            # Send the term as a new message to the chat
            chat_interface.send(
                ChatMessage(
                    term,
                    user="Terminoloog",
                    show_reaction_icons=False,
                    show_copy_icon=False,
                ),
                respond=True,
            )
        return handler
    
    # Chat callback - calls backend API
    async def chat_callback(contents: str, user: str, instance):
        """Handle chat messages by calling the backend API."""
        try:
            mode = "parallel" if filter_state.parallel_mode else "single"
            if filter_state.expand_query:
                mode += "+expansion"
            if filter_state.expand_context:
                mode += "+context"
            if filter_state.use_reranking:
                mode += "+reranking"
            logger.info(f"Sending query to backend: {contents} (mode: {mode}, prompt_set: {filter_state.prompt_set_id}, debug: {filter_state.debug_mode}, categories: {filter_state.output_categories})")
            result = await backend_client.chat(
                query=contents,
                limit=filter_state.limit,
                files=filter_state.files if filter_state.files else None,
                only_valid=filter_state.only_valid,
                debug=filter_state.debug_mode,
                parallel=filter_state.parallel_mode,
                expand_query=filter_state.expand_query,
                expand_context=filter_state.expand_context,
                use_reranking=filter_state.use_reranking,
                output_categories=filter_state.output_categories,
            )
            
            response = result["response"]
            
            # Add debug info if present
            if filter_state.debug_mode and result.get("debug_info"):
                debug_output = _format_debug_output(result["debug_info"])
                response = response + "\n\n" + debug_output
            
            # Extract See Also terms from term_entry if present
            see_also_terms = []
            if result.get("term_entry") and result["term_entry"].get("see_also"):
                see_also_terms = result["term_entry"]["see_also"]
            
            # If we have See Also terms, create a ChatMessage with footer buttons
            if see_also_terms:
                # Create buttons for each See Also term - styled as pill badges
                see_also_buttons = []
                for term in see_also_terms:
                    btn = pn.widgets.Button(
                        name=term,
                        button_type="light",
                        css_classes=["see-also-btn"],
                        margin=(2, 4),
                    )
                    btn.on_click(create_see_also_handler(term, instance))
                    see_also_buttons.append(btn)
                
                # Create a row of buttons with a label
                see_also_row = pn.Row(
                    pn.pane.HTML("<span class='see-also-label'>Vaata ka:</span>"),
                    *see_also_buttons,
                    align="center",
                    margin=(10, 0, 0, 0),
                )
                
                # Return ChatMessage with footer_objects
                return ChatMessage(
                    response,
                    user="Assistent",
                    footer_objects=[see_also_row],
                    show_reaction_icons=False,
                    show_copy_icon=True,
                )
            
            return response
        except httpx.ConnectError:
            logger.error("Cannot connect to backend")
            return "❌ Ei saa ühendust serveriga. Palun proovi hiljem uuesti."
        except httpx.TimeoutException:
            logger.error("Backend timeout")
            return "⏱️ Päring võttis liiga kaua aega. Palun proovi uuesti."
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"❌ Viga: {str(e)}"

    # UI Components
    toggle = pn.widgets.ToggleIcon(
        icon="adjustments", active_icon="adjustments-off", size="4em", align="end", value=True
    )

    @param.depends(toggle.param.value, watch=True)
    def toggle_filters(show):
        filter_column.visible = show

    text_area_input = pn.widgets.TextAreaInput(
        placeholder="Otsi dokumendist", auto_grow=True, max_rows=1
    )

    def text_area_event_handler(event):
        if event.new.endswith("\n"):
            asyncio.create_task(answer(text_area_input.value_input, text_area_input))

    async def answer(contents, active_widget):
        contents = contents.strip("\n")
        active_widget.param.update({"value": "", "value_input": ""})
        ci.send(
            ChatMessage(
                contents,
                user="Terminoloog",
                show_reaction_icons=False,
                show_copy_icon=False,
            ),
            callback=True,
        )

    text_area_input.param.watch(text_area_event_handler, "value_input")

    # Chat interface
    ci = ChatInterface(
        callback_exception="verbose",
        widgets=text_area_input,
        user="Terminoloog",
        show_send=True,
        show_button_name=False,
        callback=chat_callback,
        callback_user="Assistent",
        reset_on_send=True,
        show_stop=False,
        show_rerun=False,
        show_undo=False,
        show_copy_icon=False,
        sizing_mode="stretch_width",
        reaction_icons={},
    )

    # Filter panel
    filter_column = pn.Column(
        pn.pane.HTML("<label>Vali promptide komplekt</label>"),
        filter_handler.prompt_set_selector,
        pn.pane.HTML("<label>Vali märksõnad</label>"),
        filter_handler.keyword_selector,
        pn.pane.HTML("<label>Vali dokumendid</label>"),
        filter_handler.file_selector,
        filter_handler.limit_slider,
        filter_handler.validity_checkbox,
        pn.pane.HTML("<hr style='margin: 10px 0;'>"),
        pn.pane.HTML("<label><b>Päringu kategooriad</b></label>"),
        filter_handler.output_categories_selector,
        pn.pane.HTML("<hr style='margin: 10px 0;'>"),
        pn.pane.HTML("<label><b>Arendaja valikud</b></label>"),
        filter_handler.parallel_checkbox,
        filter_handler.expand_query_checkbox,
        filter_handler.expand_context_checkbox,
        filter_handler.reranking_checkbox,
        filter_handler.debug_checkbox,
        pn.Row(
            filter_handler.apply_filters_button,
            filter_handler.refresh_choices_button,
        ),
        visible=True,
    )

    layout = pn.Row(ci, pn.Column(toggle, filter_column))
    # See Also buttons are now created natively in footer_objects of ChatMessage
    return pn.Column(layout, sizing_mode="stretch_width")
