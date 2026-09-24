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
Logger = logging.getLogger(__name__)
