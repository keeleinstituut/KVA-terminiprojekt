"""
Frontend initialization - lightweight since backend handles heavy lifting.
"""
import os
import sys

# Ensure parent directory is in path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dotenv import load_dotenv

load_dotenv(".env")

from utils import observability as obs

obs.setup("frontend")

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

with obs.event("frontend.startup", span=False) as _ev:
    from app.views.file_upload import file_upload
    from app.views.llm_view import llm_view
    from app.views.prompt_view import prompt_view
    from app.views.document_management import document_management
    _ev.add(backend_url=os.getenv("BACKEND_URL", "http://backend:8000"))

# Export everything needed by main.py
__all__ = ['file_upload', 'llm_view', 'prompt_view', 'document_management']
