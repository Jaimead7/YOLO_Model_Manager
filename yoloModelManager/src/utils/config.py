import logging
from enum import Enum
from os import getenv
from pathlib import Path

import cv2
import ultralytics  # Needed for setting log level
from dotenv import load_dotenv
from pyUtils import (ConfigFileManager, MyLogger, ProjectPathsDict,
                     save_pyutils_logs, set_pyutils_logging_level,
                     set_pyutils_logs_path)


class EnvVars(Enum):
    IMAGES_PATH = 'IMAGES_PATH'
    MODELS_PATH = 'MODELS_PATH'
    DATASETS_PATH = 'DATASETS_PATH'
    LOGGING_LVL = 'LOGGING_LVL'
    ULTRALYTICS_LOGGING_LVL = 'ULTRALYTICS_LOGGING_LVL'


_MY_PACKAGE: ProjectPathsDict = ProjectPathsDict().set_app_path(Path(__file__).parents[2])
_MY_PACKAGE[ProjectPathsDict.DIST_PATH] = _MY_PACKAGE[ProjectPathsDict.APP_PATH] / 'dist'
_MY_PACKAGE[ProjectPathsDict.CONFIG_PATH] = _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / 'config'
_MY_PACKAGE[ProjectPathsDict.CONFIG_FILE_PATH] = _MY_PACKAGE[ProjectPathsDict.CONFIG_PATH] / 'config.toml'
MY_CFG: ConfigFileManager = ConfigFileManager(_MY_PACKAGE[ProjectPathsDict.CONFIG_FILE_PATH])
load_dotenv(
    dotenv_path= _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / '.env',
    override= False
)

# DEFAULT PATHS
IMAGES_PATH: Path = Path(getenv(
    EnvVars.IMAGES_PATH.value,
    _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / 'images'
))
MODELS_PATH: Path = Path(getenv(
    EnvVars.MODELS_PATH.value,
    _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / 'models'
))
DATASETS_PATH: Path = Path(getenv(
    EnvVars.DATASETS_PATH.value,
    _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / 'datasets'
))

# CONSTANTS FROM config.toml (Only readed on start for speed)
YOLO_IMAGE_WIDTH: int = MY_CFG.model.yolo_image_input_width
YOLO_IMAGE_HEIGHT: int = MY_CFG.model.yolo_image_input_height
RESULT_BOX_MARGIN: int = MY_CFG.model.result.box_margin
RESULT_X_TOLERANCE: int = MY_CFG.model.result.x_tolerance
RESULT_Y_TOLERANCE: int = MY_CFG.model.result.y_tolerance
RESULT_BORDER_THICKNESS: int = MY_CFG.model.result.border_thickness
RESULT_FONT_SCALE: int = MY_CFG.model.result.font_scale
RESULT_TEXT_THICKNESS: int = MY_CFG.model.result.text_thickness
RESULT_CENTER_THICKNESS: int = MY_CFG.model.result.center_thickness

# LOGGING LEVELS
LOGGING_LVL: int = MyLogger.get_logging_lvl_from_env(EnvVars.LOGGING_LVL.value)
ULTRALYTICS_LOGGING_LVL: int = MyLogger.get_logging_lvl_from_env(EnvVars.ULTRALYTICS_LOGGING_LVL.value)
cv2.setLogLevel(0) #Default 3
logging.getLogger('ultralytics').setLevel(ULTRALYTICS_LOGGING_LVL)
set_pyutils_logging_level(LOGGING_LVL)

my_logger = MyLogger(
    logger_name= 'YoloModelManager',
    logging_level= LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)

def set_yolo_manager_logs_path(new_path: Path | str) -> None:
    my_logger.logs_file_path = new_path
    set_pyutils_logs_path(new_path)

def save_yolo_manager_logs(value: bool) -> None:
    my_logger.save_logs = value
    save_pyutils_logs(value)

def set_yolo_manager_logging_level(lvl: int = logging.DEBUG) -> None:
    my_logger.set_logging_level(lvl)
    set_pyutils_logging_level(lvl)

set_yolo_manager_logging_level(logging.WARNING)
set_yolo_manager_logs_path('yoloModelManager.log')
set_yolo_manager_logging_level(LOGGING_LVL)
