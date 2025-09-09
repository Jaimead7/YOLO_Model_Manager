#---------- LOAD ENV VARIABLES FIRST ----------#
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    Path(__file__).parent / 'dist' / '.env',
    override= False
)
#---------------------------------------------#

import logging
import pyUtils

pyUtils.set_pyutils_logging_level(logging.WARNING)
pyUtils.set_pyutils_logs_path('yoloModelManager.log')  #FIXME: Not working

from .src.cameras import *
from .src.filesystem import *
from .src.image import *
from .src.model import *
from .src.utils.config import *


def set_yolo_manager_logs_path(new_path: Path | str) -> None:
    my_logger.logs_file_path = new_path
    set_pyutils_logs_path(new_path)

def save_yolo_manager_logs(value: bool) -> None:
    my_logger.save_logs = value
    save_pyutils_logs(value)

def set_yolo_manager_logging_level(lvl: int = logging.DEBUG) -> None:
    my_logger.set_logging_level(lvl)
    set_pyutils_logging_level(lvl)
