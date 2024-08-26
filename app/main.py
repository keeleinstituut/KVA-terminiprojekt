import os
import sys

import panel as pn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging.config
import os

# import weave

from app.models.qdrant_upload_scheduler import QdrantScheduler
from app.views.api_view import api_view
from app.views.chat_view import chat_view
from app.views.file_upload import file_upload
from app.views.llm_view import llm_view

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# weave.init('Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas')

logging.config.fileConfig(os.getenv('LOGGER_CONFIG'), defaults={'filename': os.getenv('LOG_FILE')})
logger = logging.getLogger('app')

# Initialize scheduler for uploading data to qdrant
scheduler = QdrantScheduler.get_instance()

class TerminologyApp():

    def __init__(self):
        file_upload_area = pn.Column(file_upload())
        api_view_area = pn.Column(api_view())
        chat_view_area = pn.Column(llm_view())

        tabs = pn.Tabs(
            ('Failide üles laadimine', file_upload_area),
            ('API päringud', api_view_area),
            ('Dokumendi päringud', chat_view_area),
        )

        self.template = pn.template.VanillaTemplate(
            title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
            main=[tabs],
            sidebar_width=300
        )


def create_app():
    app = TerminologyApp()
    return app.template


if __name__ == '__main__':
    pn.serve({'/': create_app()}, port=int(os.getenv('PANEL_PORT')))
