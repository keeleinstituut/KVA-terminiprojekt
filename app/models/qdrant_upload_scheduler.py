import atexit
import logging

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.background import BackgroundScheduler

from app.controllers.qdrant_upload_controller import upload_to_qdrant

# Configure logging
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

def job_listener(event):
    if event.exception:
        logger.error('The job "upload_to_qdrant" crashed.')

class QdrantScheduler:
    _instance = None

    @staticmethod
    def get_instance():
        if QdrantScheduler._instance is None:
            logger.info('QdrantScheduler initialization')
            QdrantScheduler._instance = BackgroundScheduler()
            QdrantScheduler._instance.add_job(upload_to_qdrant, 'interval', minutes = 10, id = 'qdrant_upload_job')
            QdrantScheduler._instance.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
            QdrantScheduler._instance.start()
            # Shut down the scheduler when exiting the app
            logger.info('Scheduler initialized')
            atexit.register(lambda: QdrantScheduler._instance.shutdown(wait=False))
        return QdrantScheduler._instance
    
    @staticmethod
    def add_listener(listener, mask):
        scheduler = QdrantScheduler.get_instance()
        scheduler.add_listener(listener, mask)
