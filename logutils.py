import os
import logging

from datasets import logging as dl

#
#
#
logging.basicConfig(
    format='[ %(asctime)s %(levelname)s %(filename)s ] %(message)s',
    datefmt='%H:%M:%S',
    level=os.environ.get('PYTHONLOGLEVEL', 'WARNING').upper(),
)
logging.captureWarnings(True)
Logger = logging.getLogger(__name__)
