"""
Prompt Management View - allows users to view, edit, and create prompts within prompt sets.
"""
import asyncio
import logging
import os

import httpx
import panel as pn
import param

pn.extension()

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

# Configure logging
logger = logging.getLogger("app")


class PromptClient:
    """Client for prompt management API."""
    
    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url
        self.timeout = 30.0
    
    async def get_prompt_sets(self) -> list:
        """Get all prompt sets."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/prompt-sets")
            response.raise_for_status()
            return response.json().get("prompt_sets", [])
    
    async def get_prompt_set(self, set_id: int) -> dict:
        """Get a specific prompt set with its prompts."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/prompt-sets/{set_id}")
            response.raise_for_status()
            return response.json()
    
    async def create_prompt_set(self, name: str, description: str = None) -> dict:
        """Create a new prompt set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/prompt-sets",
                json={"name": name, "description": description}
            )
            response.raise_for_status()
            return response.json()
    
    async def duplicate_prompt_set(self, set_id: int, new_name: str) -> dict:
        """Duplicate a prompt set with all its prompts."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/prompt-sets/{set_id}/duplicate",
                json={"new_name": new_name}
            )
            response.raise_for_status()
            return response.json()
    
    async def delete_prompt_set(self, set_id: int) -> dict:
        """Delete a prompt set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(f"{self.base_url}/prompt-sets/{set_id}")
            response.raise_for_status()
            return response.json()
    
    async def get_prompt(self, set_id: int, prompt_type: str) -> dict:
        """Get a specific prompt within a set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/prompt-sets/{set_id}/prompts/{prompt_type}")
            response.raise_for_status()
            return response.json()
    
    async def update_prompt(self, set_id: int, prompt_type: str, prompt_text: str, description: str = None) -> dict:
        """Update a prompt within a set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(
                f"{self.base_url}/prompt-sets/{set_id}/prompts/{prompt_type}",
                json={"prompt_text": prompt_text, "description": description}
            )
            response.raise_for_status()
            return response.json()
    
    async def create_prompt(self, set_id: int, prompt_type: str, prompt_text: str, description: str = None) -> dict:
        """Create a new prompt within a set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/prompt-sets/{set_id}/prompts",
                json={"prompt_type": prompt_type, "prompt_text": prompt_text, "description": description}
            )
            response.raise_for_status()
            return response.json()
    
    async def delete_prompt(self, set_id: int, prompt_type: str) -> dict:
        """Delete a prompt from a set."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(f"{self.base_url}/prompt-sets/{set_id}/prompts/{prompt_type}")
            response.raise_for_status()
            return response.json()


class PromptManager(param.Parameterized):
    """Manages prompt editing UI with prompt sets."""
    
    def __init__(self, **params):
        super().__init__(**params)
        self.client = PromptClient()
        self.prompt_sets = []
        self.current_set = None
        self.current_set_id = None
        self.current_prompt = None
        
        # UI Components - Prompt Set Selection
        self.set_selector = pn.widgets.Select(
            name="Vali promptide komplekt",
            options={},
            width=400,
        )
        
        self.duplicate_set_button = pn.widgets.Button(
            name="Kopeeri komplekt",
            button_type="warning",
            width=130,
        )
        
        self.new_set_name_input = pn.widgets.TextInput(
            name="Uue komplekti nimi",
            placeholder="Sisesta nimi...",
            width=250,
            visible=False,
        )
        
        self.confirm_duplicate_button = pn.widgets.Button(
            name="Kopeeri",
            button_type="success",
            width=80,
            visible=False,
        )
        
        # UI Components - Prompt Selection within Set
        self.prompt_selector = pn.widgets.Select(
            name="Vali prompt komplektist",
            options={},
            width=400,
        )
        
        self.prompt_type_input = pn.widgets.TextInput(
            name="Prompti tüüp (unikaalne ID)",
            placeholder="nt. my_custom_prompt",
            width=400,
        )
        
        self.description_input = pn.widgets.TextInput(
            name="Kirjeldus",
            placeholder="Prompti lühikirjeldus",
            width=400,
        )
        
        self.prompt_text_input = pn.widgets.TextAreaInput(
            name="Prompti tekst",
            placeholder="Sisesta prompti tekst...",
            height=400,
            width=600,
        )
        
        self.save_button = pn.widgets.Button(
            name="Salvesta",
            button_type="primary",
            width=100,
        )
        
        self.create_new_button = pn.widgets.Button(
            name="Loo uus prompt",
            button_type="success",
            width=150,
        )
        
        self.delete_button = pn.widgets.Button(
            name="Kustuta",
            button_type="danger",
            width=100,
        )
        
        self.refresh_button = pn.widgets.Button(
            name="Värskenda",
            button_type="default",
            width=100,
        )
        
        self.status_text = pn.pane.Markdown("", width=600)
        
        # Set up callbacks
        self.set_selector.param.watch(self._on_set_selected, 'value')
        self.prompt_selector.param.watch(self._on_prompt_selected, 'value')
        self.save_button.on_click(self._on_save)
        self.create_new_button.on_click(self._on_create_new)
        self.delete_button.on_click(self._on_delete)
        self.refresh_button.on_click(self._on_refresh)
        self.duplicate_set_button.on_click(self._on_duplicate_set_clicked)
        self.confirm_duplicate_button.on_click(self._on_confirm_duplicate)
        
        # Load prompt sets on init
        asyncio.create_task(self._load_prompt_sets())
    
    async def _load_prompt_sets(self):
        """Load all prompt sets from backend."""
        try:
            self.prompt_sets = await self.client.get_prompt_sets()
            options = {}
            default_set_id = None
            for ps in self.prompt_sets:
                label = ps.get('name', 'Unknown')
                if ps.get('is_default'):
                    label += " ✓"
                    default_set_id = ps.get('id')
                prompt_count = ps.get('prompt_count', 0)
                label += f" ({prompt_count} prompti)"
                options[label] = ps.get('id')
            self.set_selector.options = options
            # Auto-select default set
            if default_set_id is not None:
                self.set_selector.value = default_set_id
            self.status_text.object = f"✅ Laaditud {len(self.prompt_sets)} promptide komplekti"
        except Exception as e:
            logger.error(f"Failed to load prompt sets: {e}")
            self.status_text.object = f"❌ Viga komplektide laadimisel: {e}"
    
    def _on_set_selected(self, event):
        """Handle prompt set selection."""
        if event.new:
            self.current_set_id = event.new
            asyncio.create_task(self._load_set_prompts(event.new))
    
    async def _load_set_prompts(self, set_id: int):
        """Load prompts for the selected set."""
        try:
            self.current_set = await self.client.get_prompt_set(set_id)
            prompts = self.current_set.get('prompts', [])
            options = {}
            for p in prompts:
                label = f"{p.get('prompt_type')} - {p.get('description', 'No description')[:50]}"
                options[label] = p.get('prompt_type')
            self.prompt_selector.options = options
            self.prompt_selector.value = None
            self._clear_form()
            self.status_text.object = f"✅ Komplekt '{self.current_set.get('name')}' laaditud ({len(prompts)} prompti)"
        except Exception as e:
            logger.error(f"Failed to load set prompts: {e}")
            self.status_text.object = f"❌ Viga: {e}"
    
    def _on_prompt_selected(self, event):
        """Handle prompt selection within set."""
        if event.new and self.current_set_id:
            asyncio.create_task(self._load_prompt_details(event.new))
    
    async def _load_prompt_details(self, prompt_type: str):
        """Load details of selected prompt."""
        try:
            prompt = await self.client.get_prompt(self.current_set_id, prompt_type)
            self.current_prompt = prompt
            self.prompt_type_input.value = prompt.get("prompt_type", "")
            self.prompt_type_input.disabled = True  # Can't change type of existing prompt
            self.description_input.value = prompt.get("description", "") or ""
            self.prompt_text_input.value = prompt.get("prompt_text", "")
            self.status_text.object = f"✅ Laaditud: {prompt_type}"
        except Exception as e:
            logger.error(f"Failed to load prompt details: {e}")
            self.status_text.object = f"❌ Viga: {e}"
    
    def _clear_form(self):
        """Clear the prompt editing form."""
        self.current_prompt = None
        self.prompt_type_input.value = ""
        self.prompt_type_input.disabled = False
        self.description_input.value = ""
        self.prompt_text_input.value = ""
    
    def _on_save(self, event):
        """Save current prompt."""
        asyncio.create_task(self._save_prompt())
    
    async def _save_prompt(self):
        """Save the current prompt."""
        if not self.current_set_id:
            self.status_text.object = "❌ Vali kõigepealt promptide komplekt"
            return
        
        prompt_type = self.prompt_type_input.value.strip()
        prompt_text = self.prompt_text_input.value.strip()
        description = self.description_input.value.strip() or None
        
        if not prompt_type:
            self.status_text.object = "❌ Prompti tüüp on kohustuslik"
            return
        
        if len(prompt_text) < 10:
            self.status_text.object = "❌ Prompti tekst peab olema vähemalt 10 tähemärki"
            return
        
        try:
            if self.current_prompt:
                # Update existing
                await self.client.update_prompt(self.current_set_id, prompt_type, prompt_text, description)
                self.status_text.object = f"✅ Prompt '{prompt_type}' salvestatud"
            else:
                # Create new
                await self.client.create_prompt(self.current_set_id, prompt_type, prompt_text, description)
                self.status_text.object = f"✅ Uus prompt '{prompt_type}' loodud"
                self.prompt_type_input.disabled = True
            
            await self._load_set_prompts(self.current_set_id)
        except httpx.HTTPStatusError as e:
            self.status_text.object = f"❌ Viga: {e.response.text}"
        except Exception as e:
            self.status_text.object = f"❌ Viga: {e}"
    
    def _on_create_new(self, event):
        """Prepare form for creating new prompt."""
        if not self.current_set_id:
            self.status_text.object = "❌ Vali kõigepealt promptide komplekt"
            return
        self._clear_form()
        self.prompt_selector.value = None
        self.status_text.object = "📝 Loo uus prompt"
    
    def _on_delete(self, event):
        """Delete current prompt."""
        asyncio.create_task(self._delete_prompt())
    
    async def _delete_prompt(self):
        """Delete the current prompt."""
        if not self.current_prompt or not self.current_set_id:
            self.status_text.object = "❌ Vali kõigepealt prompt"
            return
        
        prompt_type = self.prompt_type_input.value
        try:
            await self.client.delete_prompt(self.current_set_id, prompt_type)
            self.status_text.object = f"✅ Prompt '{prompt_type}' kustutatud"
            self._clear_form()
            await self._load_set_prompts(self.current_set_id)
        except Exception as e:
            self.status_text.object = f"❌ Viga: {e}"
    
    def _on_refresh(self, event):
        """Refresh prompt set list."""
        asyncio.create_task(self._load_prompt_sets())
    
    def _on_duplicate_set_clicked(self, event):
        """Show duplicate set form."""
        if not self.current_set_id:
            self.status_text.object = "❌ Vali kõigepealt promptide komplekt"
            return
        self.new_set_name_input.visible = True
        self.confirm_duplicate_button.visible = True
        self.new_set_name_input.value = f"{self.current_set.get('name', '')} (koopia)"
    
    def _on_confirm_duplicate(self, event):
        """Confirm duplication of prompt set."""
        asyncio.create_task(self._duplicate_set())
    
    async def _duplicate_set(self):
        """Duplicate the current prompt set."""
        new_name = self.new_set_name_input.value.strip()
        if not new_name:
            self.status_text.object = "❌ Sisesta uue komplekti nimi"
            return
        
        try:
            result = await self.client.duplicate_prompt_set(self.current_set_id, new_name)
            self.status_text.object = f"✅ {result.get('message', 'Komplekt kopeeritud')}"
            self.new_set_name_input.visible = False
            self.confirm_duplicate_button.visible = False
            await self._load_prompt_sets()
        except httpx.HTTPStatusError as e:
            self.status_text.object = f"❌ Viga: {e.response.text}"
        except Exception as e:
            self.status_text.object = f"❌ Viga: {e}"
    
    def view(self):
        """Create the prompt management view."""
        return pn.Column(
            pn.pane.Markdown("## Promptide haldamine"),
            pn.pane.Markdown("Vali promptide komplekt ja muuda selle sees olevaid prompte."),
            
            # Prompt Set selection
            pn.pane.Markdown("### Promptide komplekt"),
            pn.Row(
                self.set_selector,
                self.refresh_button,
                self.duplicate_set_button,
            ),
            pn.Row(
                self.new_set_name_input,
                self.confirm_duplicate_button,
            ),
            
            pn.layout.Divider(),
            
            # Prompt selection within set
            pn.pane.Markdown("### Promptid komplektis"),
            pn.Row(
                self.prompt_selector,
                self.create_new_button,
            ),
            
            pn.layout.Divider(),
            
            # Prompt editing
            self.prompt_type_input,
            self.description_input,
            self.prompt_text_input,
            pn.Row(
                self.save_button,
                self.delete_button,
            ),
            self.status_text,
            sizing_mode="stretch_width",
        )


def prompt_view():
    """Create and return the prompt management view."""
    manager = PromptManager()
    return manager.view()
