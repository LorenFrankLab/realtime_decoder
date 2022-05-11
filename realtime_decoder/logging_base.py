import os
import logging

class LoggingClass(object):
    """A class that logs information
    """
    def __init__(self):
        self.class_log = logging.getLogger(
            name=f'{self.__class__.__name__}'
        )

class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)