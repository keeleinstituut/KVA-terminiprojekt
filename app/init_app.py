"""
Frontend initialization - lightweight since backend handles heavy lifting.
"""
import os
import sys
import logging.config

# Ensure parent directory is in path for imports
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dotenv import load_dotenv

print("=" * 50, flush=True)
print("KVA Frontend - Starting up...", flush=True)
print("=" * 50, flush=True)

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Loading environment variables...", flush=True)
load_dotenv(".env")

# Initialize logging configuration
print("Initializing logging...", flush=True)
logging.config.fileConfig(os.getenv("LOGGER_CONFIG"))
logger = logging.getLogger("app")

print("Loading views...", flush=True)
from app.views.file_upload import file_upload
from app.views.llm_view import llm_view
from app.views.prompt_view import prompt_view
from app.views.document_management import document_management

print("Frontend initialization complete!", flush=True)
print("=" * 50, flush=True)

# Export everything needed by main.py
__all__ = ['logger', 'file_upload', 'llm_view', 'prompt_view', 'document_management']
