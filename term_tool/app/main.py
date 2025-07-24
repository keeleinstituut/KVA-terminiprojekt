import os
import sys

import panel as pn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging.config
import os

from dotenv import load_dotenv

from app.models.qdrant_upload_scheduler import QdrantScheduler
from app.views.api_view import api_view
from app.views.file_upload import file_upload
from app.views.llm_view import llm_view
from utils import iate_api_helpers

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv(".env")

# Initialize logging configuration
logging.config.fileConfig(
    os.getenv("LOGGER_CONFIG"), defaults={"filename": os.getenv("LOG_FILE")}
)
logger = logging.getLogger("app")

# Initialize scheduler for uploading data to qdrant
scheduler = QdrantScheduler.get_instance()

# Initialize domains (assuming this is necessary for API operations)
domains = iate_api_helpers.initialize_domains()


# Define the layout areas for the Panel UI
def create_file_upload_area():
    return pn.Column(file_upload())


def create_api_view_area():
    return pn.Column(api_view(domains))


def create_chat_view_area():
    return pn.Column(llm_view())


# Create the template and structure for the UI
def create_template(guest_mode=False):
    try:
        file_upload_area = create_file_upload_area()
        logger.info("File upload area created successfully")
    except Exception as e:
        logger.error(f"Error creating file upload area: {e}")
        file_upload_area = pn.Column(
            pn.pane.Markdown("**File upload is currently unavailable.**")
        )
    
    try:
        api_view_area = create_api_view_area()
        logger.info("API view area created successfully")
    except Exception as e:
        logger.error(f"Error creating API view area: {e}")
        api_view_area = pn.Column(
            pn.pane.Markdown("**API view is currently unavailable.**")
        )
    try:
        chat_view_area = create_chat_view_area()
        logger.info("Chat view area created successfully")
    except Exception as e:
        logger.error(f"Error creating chat view area: {e}")
        chat_view_area = pn.Column(
            pn.pane.Markdown("**Chat view is currently unavailable.**")
        )

    tabs = pn.Tabs(
       ("API päringud", api_view_area),
       ("Dokumendiotsing", chat_view_area),
    )

    if not guest_mode:
      #  pass
        tabs.append(("Failide üleslaadimine", file_upload_area))
    
    template = pn.template.VanillaTemplate(
        title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
        main=[tabs],
        sidebar_width=300,
    )
    
    logger.info("Template created successfully")

    return template


# Main function to create and serve the Panel app
def create_app(guest_mode: bool = False):
    template = create_template(guest_mode=guest_mode)
    return template


username = pn.state.user

logger.info(pn.state.user_info)

if username == "kylaline":
    template = create_app(guest_mode=True).servable()
else:
    template = create_app().servable()
