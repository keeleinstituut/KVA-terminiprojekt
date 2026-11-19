"""
Prompt Management View - allows users to view and edit prompts.
Simplified to work with a single default prompt set.
"""
import asyncio

import panel as pn
import param

from app.telemetry import backend_client, describe_http_error, ui_event, ui_log

pn.extension()



class PromptClient:
    """Client for prompt management API."""
    
    def __init__(self):
        self.timeout = 30.0
    
    async def get_default_set_id(self) -> int:
        """Get the default prompt set ID."""
        async with backend_client(timeout=self.timeout) as client:
            response = await client.get(f"/prompt-sets")
            response.raise_for_status()
            prompt_sets = response.json().get("prompt_sets", [])
            for ps in prompt_sets:
                if ps.get("is_default"):
                    return ps.get("id")
            if prompt_sets:
                return prompt_sets[0].get("id")
            return None
    
    async def get_prompts(self, set_id: int) -> list:
        """Get all prompts for the default set."""
        async with backend_client(timeout=self.timeout) as client:
            response = await client.get(f"/prompt-sets/{set_id}")
            response.raise_for_status()
            return response.json().get("prompts", [])
    
    async def get_prompt(self, set_id: int, prompt_type: str) -> dict:
        """Get a specific prompt."""
        async with backend_client(timeout=self.timeout) as client:
            response = await client.get(f"/prompt-sets/{set_id}/prompts/{prompt_type}")
            response.raise_for_status()
            return response.json()
    
    async def update_prompt(self, set_id: int, prompt_type: str, prompt_text: str, description: str = None) -> dict:
        """Update a prompt."""
        async with backend_client(timeout=self.timeout) as client:
            response = await client.put(
                f"/prompt-sets/{set_id}/prompts/{prompt_type}",
                json={"prompt_text": prompt_text, "description": description}
            )
            response.raise_for_status()
            return response.json()


class PromptManager(param.Parameterized):
    """Manages prompt editing UI."""
    
    def __init__(self, **params):
        super().__init__(**params)
        self.client = PromptClient()
        self.default_set_id = None
        self.current_prompt_type = None
        
        # UI Components
        self.prompt_selector = pn.widgets.Select(
            name="Vali prompt",
            options={},
            width=400,
        )
        
        self.prompt_text_input = pn.widgets.TextAreaInput(
            name="Prompti tekst",
            placeholder="Vali prompt muutmiseks...",
            height=500,
            width=800,
        )
        
        self.save_button = pn.widgets.Button(
            name="Salvesta",
            button_type="primary",
            width=100,
        )
        
        self.status_text = pn.pane.Markdown("", width=800)
        
        # Set up callbacks
        self.prompt_selector.param.watch(self._on_prompt_selected, 'value')
        self.save_button.on_click(self._on_save)
        
        # Load prompts on init
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize by loading the default set and its prompts."""
        try:
            self.default_set_id = await self.client.get_default_set_id()
            if self.default_set_id is None:
                self.status_text.object = "❌ Viga: Promptide komplekti ei leitud"
                return
            
            prompts = await self.client.get_prompts(self.default_set_id)
            
            # Define logical sort order for prompts
            # Extraction prompts first, then category-specific query expansion prompts
            sort_order = [
                'definitions_extraction',          # Extraction prompts
                'related_terms_extraction',
                'usage_evidence_extraction',
                'see_also_extraction',
                'definitions_query_expansion',     # Category-specific query expansions
                'related_terms_query_expansion',
                'usage_evidence_query_expansion',
            ]
            
            # Create ordered dictionary
            options = {}
            prompt_types = [p.get('prompt_type') for p in prompts]
            
            # Add prompts in sort order
            for prompt_type in sort_order:
                if prompt_type in prompt_types:
                    options[prompt_type] = prompt_type
            
            # Add any remaining prompts not in sort_order (for future compatibility)
            for prompt_type in sorted(prompt_types):
                if prompt_type not in options:
                    options[prompt_type] = prompt_type
            
            self.prompt_selector.options = options
            self.status_text.object = ""
            
        except Exception as e:
            # type is recorded too: this used to log an empty message
            ui_log("prompts.init", outcome="error", error=describe_http_error(e))
            self.status_text.object = f"❌ Viga: {e}"
    
    def _on_prompt_selected(self, event):
        """Handle prompt selection."""
        if event.new:
            self.current_prompt_type = event.new
            asyncio.create_task(self._load_prompt(event.new))
    
    async def _load_prompt(self, prompt_type: str):
        """Load prompt text."""
        try:
            prompt = await self.client.get_prompt(self.default_set_id, prompt_type)
            self.prompt_text_input.value = prompt.get("prompt_text", "")
            self.status_text.object = ""
        except Exception as e:
            ui_log("prompts.load", outcome="error", prompt_type=prompt_type,
                   error=describe_http_error(e))
            self.status_text.object = f"❌ Viga: {e}"
    
    def _on_save(self, event):
        """Save current prompt."""
        asyncio.create_task(self._save_prompt())
    
    async def _save_prompt(self):
        """Save the current prompt."""
        if not self.current_prompt_type:
            self.status_text.object = "❌ Vali kõigepealt prompt"
            return
        
        prompt_text = self.prompt_text_input.value.strip()
        
        if len(prompt_text) < 10:
            self.status_text.object = "❌ Prompti tekst peab olema vähemalt 10 tähemärki"
            return
        
        with ui_event("prompt.update", prompt_type=self.current_prompt_type,
                      prompt_set_id=self.default_set_id, prompt_chars=len(prompt_text)) as ev:
            try:
                result = await self.client.update_prompt(
                    self.default_set_id,
                    self.current_prompt_type,
                    prompt_text
                )

                if result.get("success"):
                    self.status_text.object = f"✅ Salvestatud: {self.current_prompt_type}"
                else:
                    ev.set_outcome("error")
                    ev.add(backend_message=result.get("message"))
                    self.status_text.object = f"❌ Viga: {result.get('message', 'Tundmatu viga')}"

            except Exception as e:
                ev.set_error(e)
                ev.add(backend_error=describe_http_error(e))
                self.status_text.object = f"❌ Viga: {e}"
    
    def view(self):
        """Create the prompt management view."""
        return pn.Column(
            self.prompt_selector,
            self.prompt_text_input,
            pn.Row(
                self.save_button,
                self.status_text,
            ),
            sizing_mode="stretch_width",
        )


def prompt_view():
    """Create and return the prompt management view."""
    manager = PromptManager()
    return manager.view()
