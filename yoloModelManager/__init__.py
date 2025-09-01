from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    Path(__file__).parent / 'dist' / '.env',
    override= False
)

import pyUtils

pyUtils.set_pyutils_logs_path('yoloModelManager.log')
pyUtils.save_pyutils_logs(True)

from .src.cameras import *
from .src.cameras.camera_manager import my_logger as camera_manager_logger
from .src.filesystem import *
from .src.filesystem.dirs import my_logger as dirs_logger
from .src.filesystem.dirs_managers import my_logger as dirs_managers_logger
from .src.filesystem.files import my_logger as files_logger
from .src.image import *
from .src.image.image_processing import my_logger as image_processing_logger
from .src.model import *
from .src.model.data import my_logger as model_data_logger
from .src.model.model_manager import my_logger as model_manager_logger
from .src.scripts.camera import my_logger as scripts_camera_logger
from .src.scripts.dataset import my_logger as scripts_dataset_logger
from .src.scripts.model import my_logger as scripts_model_logger
from .src.utils import *

loggers: tuple[MyLogger, ...] = (
    camera_manager_logger,
    dirs_logger,
    dirs_managers_logger,
    files_logger,
    image_processing_logger,
    model_data_logger,
    model_manager_logger,
    scripts_camera_logger,
    scripts_dataset_logger,
    scripts_model_logger
)

def set_yolo_manager_logs_path(new_path: Path | str) -> None:
    for logger in loggers:
        logger.logs_file_path = new_path
    pyUtils.set_pyutils_logs_path(new_path)

def save_yolo_manager_logs(value: bool) -> None:
    for logger in loggers:
        logger.save_logs = value
    pyUtils.save_pyutils_logs(value)

def set_yolo_manager_logging_level(lvl: int = logging.DEBUG) -> None:
    for logger in loggers:
        logger.set_logging_level(lvl)
    pyUtils.set_pyutils_logging_level(lvl)
