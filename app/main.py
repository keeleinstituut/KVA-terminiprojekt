import sys
import os
import panel as pn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from apscheduler.schedulers.background import BackgroundScheduler
from app.views.api_view import api_view
from app.views.file_upload import file_upload
from app.views.chat_view import chat_view
import logging.config

from app.controllers.qdrant_upload_controller import upload_to_qdrant

logging.config.fileConfig('../config/logging.config')
logger = logging.getLogger('app')

def job_listener(event):
    if event.exception:
        logger.error('The job "upload_to_qdrant" crashed.')

scheduler = BackgroundScheduler()
scheduler.add_job(upload_to_qdrant, 'interval', minutes = 10, id = 'qdrant_upload_job')
scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
scheduler.start()

file_upload_area = pn.Column(file_upload())
api_view_area = pn.Column(api_view())
chat_view_area = pn.Column(chat_view())

tabs = pn.Tabs(
    ('Failide üles laadimine', file_upload_area),
    ('API päringud', api_view_area),
    ('Dokumendi päringud', chat_view_area),
)

template = pn.template.VanillaTemplate(
    title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas",
    main=[tabs],
    sidebar_width=300
)

pn.serve(template, title="Tehisintellekti rakendamine riigikaitseterminoloogia valdkonnas")
