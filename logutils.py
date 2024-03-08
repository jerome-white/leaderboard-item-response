import os
import logging

logging.basicConfig(
    format='[ %(asctime)s %(levelname)s %(filename)s ] %(message)s',
    datefmt='%H:%M:%S',
    level=os.environ.get('PYTHONLOGLEVEL', 'WARNING').upper(),
)
logging.captureWarnings(True)
# logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
Logger = logging.getLogger(__name__)
