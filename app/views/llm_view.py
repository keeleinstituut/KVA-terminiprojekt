"""
LLM View - Panel frontend that calls the FastAPI backend.
No heavy model loading here - all processing done by backend.
"""
import asyncio
import json
import logging
import os

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

# Load config.json to get reranking setting
def load_reranking_config():
    """Load reranking enabled status from config.json"""
    # Try multiple possible paths
    possible_paths = [
        os.getenv('APP_CONFIG'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.json'),
        '/app/config/config.json',  # Docker path
        os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json'),
    ]
    
    for config_path in possible_paths:
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    enabled = config.get('reranking', {}).get('enabled', False)
                    logger.info(f"Loaded reranking config from {config_path}: enabled={enabled}")
                    return enabled
            except Exception as e:
                logger.warning(f"Failed to load config.json from {config_path}: {e}")
                continue
    
    logger.warning("Could not find config.json, defaulting reranking to False")
    return False

RERANKING_ENABLED = load_reranking_config()


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
        expand_query: bool = True,
        expand_context: bool = False,
        use_reranking: bool = True,
        output_categories: list = None,
        early_parallelization: bool = True,
    ) -> dict:
        """Call the chat endpoint (always uses parallel extraction mode)."""
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
                    "expand_query": expand_query,
                    "early_parallelization": early_parallelization,
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


class FilterState:
    """
    Manages filter state for the chat.
    """
    def __init__(self):
        self.files: list = []
        self.limit: int = 8  # Increased from 5 for better term diversity
        self.only_valid: bool = False
        self.debug_mode: bool = False
        self.parallel_mode: bool = True  # Always enabled
        self.expand_query: bool = True   # Always enabled
        self.expand_context: bool = False  # Always disabled
        self.use_reranking: bool = RERANKING_ENABLED  # From config.json
        self.early_parallelization: bool = True  # Always enabled
        self.output_categories: list = ["definitions", "related_terms", "usage_evidence"]  # All enabled by default
    
    def apply(self, files: list, limit: int, only_valid: bool, debug_mode: bool = False, expand_query: bool = True, expand_context: bool = False, use_reranking: bool = True, output_categories: list = None, early_parallelization: bool = True):
        self.files = files
        self.limit = limit
        self.only_valid = only_valid
        self.debug_mode = debug_mode
        # parallel_mode is always True (removed as parameter)
        self.expand_query = expand_query
        self.expand_context = expand_context
        self.use_reranking = use_reranking
        self.early_parallelization = early_parallelization
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
        
        self.debug_checkbox = pn.widgets.Checkbox(
            name="🔍 Debug mode (näita pipeline'i samme)", 
            width=500,
            value=False,
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
            debug_mode=self.debug_checkbox.value,
            expand_query=True,  # Always enabled
            expand_context=False,  # Always disabled
            use_reranking=RERANKING_ENABLED,  # From config.json
            output_categories=self.output_categories_selector.value,
            early_parallelization=True,  # Always enabled
        )
        logger.info(f"Filters applied: {self.filter_state.__dict__}")


def _format_debug_output(debug_info: dict) -> str:
    """Format debug information to show the pipeline flow clearly."""
    if not debug_info:
        return ""
    
    import json
    
    extraction_mode = debug_info.get('extraction_mode', 'single')
    total_duration = debug_info.get('total_duration_ms', 0)
    search_type = debug_info.get('search_type', 'unknown')
    model_used = debug_info.get('model_used', 'unknown')
    
    steps = debug_info.get("pipeline_steps", [])
    
    # Parse step data
    query = ""
    filters_info = {}
    expansion_data = {}  # category -> {expanded_terms, chunks, duration}
    extraction_results = {}  # category -> {items, count, duration}
    
    for step in steps:
        name = step.get("step_name", "")
        data_str = step.get("data", "")
        
        if not data_str:
            continue
            
        try:
            data = json.loads(data_str)
        except:
            continue
        
        if "Query Received" in name:
            query = data.get("query", "")
        
        if "Filters Applied" in name:
            filters_info = data
        
        if "Per-Category" in name:
            for cat, cat_data in data.items():
                if isinstance(cat_data, dict):
                    expansion_data[cat] = {
                        "expanded_terms": cat_data.get("expanded_terms", []),
                        "chunks": cat_data.get("chunks", []),
                        "duration_ms": cat_data.get("duration_ms", 0),
                    }
        
        if "Parallel LLM" in name:
            for ext in data:
                if isinstance(ext, dict) and "extraction" in ext:
                    cat = ext.get("extraction")
                    extraction_results[cat] = {
                        "items": ext.get("extracted_items", []),
                        "count": ext.get("items_found", 0),
                        "duration_ms": ext.get("duration_ms", 0),
                        "chunks_available": ext.get("chunks_available", 0),
                    }
    
    lines = []
    lines.append("---")
    lines.append(f"## 🔍 Pipeline Debug: \"{query}\"")
    lines.append(f"*Kokku: {total_duration:.0f}ms | {extraction_mode} | {model_used}*")
    lines.append("")
    
    # Step 1 & 2: Query and filters
    lines.append("---")
    lines.append("### 1️⃣ Päring ja filtrid")
    lines.append(f"- **Märksõna:** {query}")
    lines.append(f"- **Lõikude piirang:** {filters_info.get('limit', '?')}")
    lines.append(f"- **Režiim:** {extraction_mode}")
    lines.append("")
    
    # Step 3: Query expansion and search (per category)
    if expansion_data:
        lines.append("---")
        lines.append("### 2️⃣ Päringu laiendamine ja otsing")
        lines.append("")
        
        for cat, cat_info in expansion_data.items():
            cat_label = {"definitions": "📖 Definitsioonid", "related_terms": "🔗 Seotud terminid",
                         "usage_evidence": "📝 Kasutusseosed"}.get(cat, cat)
            
            expanded = cat_info.get("expanded_terms", [])
            chunks = cat_info.get("chunks", [])
            duration = cat_info.get("duration_ms", 0)
            
            lines.append(f"#### {cat_label}")
            lines.append(f"*{duration:.0f}ms*")
            lines.append("")
            
            if expanded:
                lines.append(f"**Laiendatud otsing:** {', '.join(expanded)}")
                lines.append("")
            
            lines.append(f"**Leitud {len(chunks)} lõiku:**")
            lines.append("")
            
            for i, chunk in enumerate(chunks, 1):  # Show all chunks
                title = chunk.get("title", "?")
                page = chunk.get("page", "?")
                score = chunk.get("score", 0)
                lines.append(f"{i}. *{title}* (lk {page}, skoor {score:.2f})")
            lines.append("")
            
            # Expandable full chunks
            if chunks:
                lines.append("<details>")
                lines.append("<summary>Näita lõikude sisu</summary>")
                lines.append("")
                for i, chunk in enumerate(chunks, 1):
                    title = chunk.get("title", "?")
                    page = chunk.get("page", "?")
                    text = chunk.get("text_full", chunk.get("text_preview", ""))
                    lines.append(f"**{i}. {title} (lk {page})**")
                    lines.append("```")
                    lines.append(text)  # Full text, no truncation
                    lines.append("```")
                    lines.append("")
                lines.append("</details>")
                lines.append("")
    
    # Step 4: LLM extraction results
    if extraction_results:
        lines.append("---")
        lines.append("### 3️⃣ LLM töötlus ja tulemused")
        lines.append("")
        
        for cat, cat_info in extraction_results.items():
            cat_label = {"definitions": "📖 Definitsioonid", "related_terms": "🔗 Seotud terminid",
                         "usage_evidence": "📝 Kasutusseosed", "see_also": "👁️ Vaata ka"}.get(cat, cat)
            
            items = cat_info.get("items", [])
            count = cat_info.get("count", 0)
            duration = cat_info.get("duration_ms", 0)
            chunks_available = cat_info.get("chunks_available", 0)
            
            lines.append(f"#### {cat_label}")
            lines.append(f"*{chunks_available} lõiku → {count} tulemust ({duration:.0f}ms)*")
            lines.append("")
            
            if items:
                for i, item in enumerate(items, 1):  # Show all items
                    if isinstance(item, dict):
                        text = item.get("text", item.get("term", str(item)))
                        source = item.get("source", "")
                        page = item.get("page", "")
                        if source:
                            lines.append(f"{i}. ✅ {text}")
                            lines.append(f"   *← {source}, lk {page}*")
                        else:
                            lines.append(f"{i}. ✅ {text}")
                    else:
                        lines.append(f"{i}. ✅ {item}")
            else:
                lines.append("*(midagi ei leitud)*")
            
            lines.append("")
    
    # Step 5: Source-result mapping (which chunks produced results)
    if expansion_data and extraction_results:
        lines.append("---")
        lines.append("### 4️⃣ Lõikude kasutamine")
        lines.append("*Millised lõigud andsid tulemusi?*")
        lines.append("")
        
        # For each category, show which chunks were used
        for cat in ["definitions", "related_terms", "usage_evidence"]:
            if cat not in expansion_data or cat not in extraction_results:
                continue
            
            cat_label = {"definitions": "📖 Definitsioonid", "related_terms": "🔗 Seotud terminid",
                         "usage_evidence": "📝 Kasutusseosed"}.get(cat, cat)
            
            chunks = expansion_data[cat].get("chunks", [])
            items = extraction_results[cat].get("items", [])
            
            # Build lookup
            used_sources = set()
            for item in items:
                if isinstance(item, dict):
                    source = item.get("source", "")
                    page = item.get("page", "")
                    used_sources.add(f"{source}|{page}")
            
            used_count = 0
            lines.append(f"**{cat_label}:**")
            
            for chunk in chunks:  # Show all chunks
                title = chunk.get("title", "?")
                page = chunk.get("page", "?")
                key = f"{title}|{page}"
                
                if key in used_sources:
                    lines.append(f"- ✅ {title} (lk {page})")
                    used_count += 1
                else:
                    lines.append(f"- ⬜ {title} (lk {page})")
            
            efficiency = f"{(used_count/len(chunks)*100):.0f}%" if chunks else "0%"
            lines.append(f"  **Kasutatud: {used_count}/{len(chunks)} ({efficiency})**")
            lines.append("")
    
    # LLM Call Details - Show full inputs and outputs (collapsible)
    lines.append("---")
    lines.append("### 🔬 LLM Kõned (täielik sisend ja väljund)")
    lines.append("")
    
    # Find all LLM-related steps and group them by call
    llm_steps = []
    for step in steps:
        step_name = step.get("step_name", "")
        if "LLM" in step_name or "Query Expansion" in step_name:
            llm_steps.append(step)
    
    # Group input/output pairs together
    if llm_steps:
        # Group steps by their base name (e.g., "Query Expansion LLM Input [definitions]" pairs with "Query Expansion LLM Output [definitions]")
        llm_calls = {}
        for step in llm_steps:
            step_name = step.get("step_name", "")
            # Extract the call identifier (e.g., "[definitions]" or "[generic]")
            import re
            match = re.search(r'\[([^\]]+)\]', step_name)
            call_id = match.group(1) if match else "unknown"
            
            # Determine if it's input or output
            if "Input" in step_name:
                call_key = f"{call_id}_input"
            elif "Output" in step_name or "Response" in step_name:
                call_key = f"{call_id}_output"
            else:
                call_key = f"{call_id}_other"
            
            if call_id not in llm_calls:
                llm_calls[call_id] = {}
            llm_calls[call_id][call_key] = step
        
        # Render each LLM call as a collapsible section
        for call_id, call_steps in sorted(llm_calls.items()):
            input_step = call_steps.get(f"{call_id}_input")
            output_step = call_steps.get(f"{call_id}_output")
            
            # Create a summary title
            if input_step:
                step_name = input_step.get("step_name", "")
                duration = input_step.get("duration_ms", 0)
                if output_step:
                    duration += output_step.get("duration_ms", 0)
            elif output_step:
                step_name = output_step.get("step_name", "")
                duration = output_step.get("duration_ms", 0)
            else:
                continue
            
            # Extract call type from step name
            call_type = "LLM Kõne"
            if "Query Expansion" in step_name:
                call_type = f"Päringu laiendamine [{call_id}]"
            elif "Parallel LLM" in step_name:
                call_type = f"Paralleelne ekstraktsioon [{call_id}]"
            
            lines.append("<details>")
            lines.append(f"<summary><b>{call_type}</b> ({duration:.0f}ms)</summary>")
            lines.append("")
            
            # Show input
            if input_step:
                data_str = input_step.get("data", "")
                if data_str:
                    try:
                        data = json.loads(data_str)
                        lines.append("**📥 LLM Sisend (täielik prompt):**")
                        lines.append("")
                        if isinstance(data, dict):
                            if "full_prompt" in data:
                                lines.append("```")
                                lines.append(data["full_prompt"])
                                lines.append("```")
                            elif "full_prompt_text" in data:
                                lines.append("```")
                                lines.append(data["full_prompt_text"])
                                lines.append("```")
                            elif "messages" in data:
                                for msg in data["messages"]:
                                    lines.append(f"**{msg.get('role', 'unknown')}:**")
                                    lines.append("```")
                                    lines.append(msg.get("content", ""))
                                    lines.append("```")
                                    lines.append("")
                            elif "system_prompt" in data:
                                lines.append("**Süsteemi prompt:**")
                                lines.append("```")
                                lines.append(data["system_prompt"])
                                lines.append("```")
                                lines.append("")
                                if "query" in data:
                                    lines.append(f"**Päring:** {data['query']}")
                                    lines.append("")
                                if "context" in data or "context_preview" in data:
                                    ctx = data.get("context") or data.get("context_preview", "")
                                    lines.append("**Kontekst:**")
                                    lines.append("```")
                                    lines.append(ctx)
                                    lines.append("```")
                                    lines.append("")
                        else:
                            lines.append("```")
                            lines.append(str(data))
                            lines.append("```")
                        lines.append("")
                    except:
                        lines.append("```")
                        lines.append(data_str)
                        lines.append("```")
                        lines.append("")
            
            # Show output
            if output_step:
                data_str = output_step.get("data", "")
                if data_str:
                    try:
                        data = json.loads(data_str)
                        lines.append("**📤 LLM Väljund (täielik vastus):**")
                        lines.append("")
                        if isinstance(data, dict):
                            raw_response = data.get("raw_response", data.get("response", str(data)))
                            lines.append("```")
                            lines.append(raw_response)
                            lines.append("```")
                            lines.append("")
                            if "response_length" in data:
                                lines.append(f"*Pikkus: {data['response_length']} tähemärki*")
                                lines.append("")
                        else:
                            lines.append("```")
                            lines.append(str(data))
                            lines.append("```")
                            lines.append("")
                    except:
                        lines.append("```")
                        lines.append(data_str)
                        lines.append("```")
                        lines.append("")
            
            lines.append("</details>")
            lines.append("")
    
    # All pipeline steps (collapsed)
    lines.append("---")
    lines.append("<details>")
    lines.append("<summary>📋 Kõik pipeline sammud (täielik)</summary>")
    lines.append("")
    
    for step in steps:
        step_num = step.get("step_number", 0)
        step_name = step.get("step_name", "Unknown")
        duration = step.get("duration_ms", 0)
        data_str = step.get("data", "")
        
        lines.append(f"**Step {step_num}: {step_name}** ({duration:.0f}ms)")
        if data_str:
            lines.append("<details>")
            lines.append("<summary>Näita andmeid</summary>")
            lines.append("")
            lines.append("```json")
            # Show full data, no truncation
            lines.append(data_str)
            lines.append("```")
            lines.append("</details>")
        lines.append("")
    
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
            # Always parallel mode now
            mode = "parallel"
            if filter_state.expand_query:
                mode += "+per-category-expansion" if filter_state.early_parallelization else "+single-expansion"
            if filter_state.expand_context:
                mode += "+context"
            if filter_state.use_reranking:
                mode += "+reranking"
            logger.info(f"Sending query to backend: {contents} (mode: {mode}, debug: {filter_state.debug_mode}, categories: {filter_state.output_categories})")
            result = await backend_client.chat(
                query=contents,
                limit=filter_state.limit,
                files=filter_state.files if filter_state.files else None,
                only_valid=filter_state.only_valid,
                debug=filter_state.debug_mode,
                expand_query=filter_state.expand_query,
                expand_context=filter_state.expand_context,
                use_reranking=filter_state.use_reranking,
                early_parallelization=filter_state.early_parallelization,
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
