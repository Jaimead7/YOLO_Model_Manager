import logging
from enum import Enum
from os import getenv
from pathlib import Path

import cv2
import ultralytics  # Needed for setting log level
from dotenv import load_dotenv
from pyUtils import (ConfigFileManager, MyLogger, ProjectPathsDict,
                     set_pyutils_logging_level)


class EnvVars(Enum):
    IMAGES_PATH = 'IMAGES_PATH'
    MODELS_PATH = 'MODELS_PATH'
    DATASETS_PATH = 'DATASETS_PATH'
    PYUTILS_LOGGING_LVL = 'PYUTILS_LOGGING_LVL'
    FILESYSTEM_LOGGING_LVL = 'FILESYSTEM_LOGGING_LVL'
    SCRIPTS_LOGGING_LVL = 'SCRIPTS_LOGGING_LVL'
    IMAGE_LOGGING_LVL = 'IMAGE_LOGGING_LVL'
    MODEL_LOGGING_LVL = 'MODEL_LOGGING_LVL'
    CAMERA_LOGGING_LVL = 'CAMERA_LOGGING_LVL'
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
#TODO: change to pyUtils
def get_logging_lvl_from_env(env_var_name: str) -> int:
    env_var: str | int = getenv(env_var_name, logging.DEBUG)
    try:
        return int(env_var)
    except ValueError:
        return MyLogger.get_lvl_int(str(env_var))
PYUTILS_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.PYUTILS_LOGGING_LVL.value)
FILESYSTEM_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.FILESYSTEM_LOGGING_LVL.value)
SCRIPTS_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.SCRIPTS_LOGGING_LVL.value)
IMAGE_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.IMAGE_LOGGING_LVL.value)
MODEL_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.MODEL_LOGGING_LVL.value)
CAMERA_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.CAMERA_LOGGING_LVL.value)
ULTRALYTICS_LOGGING_LVL: int = get_logging_lvl_from_env(EnvVars.ULTRALYTICS_LOGGING_LVL.value)
cv2.setLogLevel(0) #Default 3
logging.getLogger('ultralytics').setLevel(ULTRALYTICS_LOGGING_LVL)
set_pyutils_logging_level(PYUTILS_LOGGING_LVL)
