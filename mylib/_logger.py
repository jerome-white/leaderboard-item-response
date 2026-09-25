import os
import logging

#
#
#
logging.basicConfig(
    format='[ %(asctime)s %(levelname)s %(filename)s ] %(message)s',
    datefmt='%H:%M:%S',
    level=os.environ.get('PYTHONLOGLEVEL', 'INFO').upper(),
)
logging.captureWarnings(True)
logging.getLogger('httpx').setLevel(logging.WARNING)
Logger = logging.getLogger(__name__)
