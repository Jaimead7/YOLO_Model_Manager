import logging
from enum import Enum
from os import getenv
from pathlib import Path

import cv2
import ultralytics  # Needed for setting log level
from dotenv import load_dotenv
from pyUtils import ConfigFileManager, MyLogger, ProjectPathsDict


class EnvVars(Enum):
    IMAGES_PATH = 'IMAGES_PATH'
    MODELS_PATH = 'MODELS_PATH'
    DATASETS_PATH = 'DATASETS_PATH'
    TIMING_LOGGING_LVL = 'TIMING_LOGGING_LVL'
    FILESYSTEM_LOGGING_LVL = 'FILESYSTEM_LOGGING_LVL'
    SCRIPTS_LOGGING_LVL = 'SCRIPTS_LOGGING_LVL'
    IMAGE_LOGGING_LVL = 'IMAGE_LOGGING_LVL'
    MODEL_LOGGING_LVL = 'MODEL_LOGGING_LVL'
    CAMERA_LOGGING_LVL = 'CAMERA_LOGGING_LVL'

_MY_PACKAGE: ProjectPathsDict = ProjectPathsDict().set_app_path(Path(__file__).parents[2])
_MY_PACKAGE[ProjectPathsDict.DIST_PATH] = _MY_PACKAGE[ProjectPathsDict.APP_PATH] / 'dist'
_MY_PACKAGE[ProjectPathsDict.CONFIG_PATH] = _MY_PACKAGE[ProjectPathsDict.DIST_PATH] / 'config'
_MY_PACKAGE[ProjectPathsDict.CONFIG_FILE_PATH] = _MY_PACKAGE[ProjectPathsDict.CONFIG_PATH] / 'config.toml'
_MY_CFG: ConfigFileManager = ConfigFileManager(_MY_PACKAGE[ProjectPathsDict.CONFIG_FILE_PATH])
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

# CONSTANTS
YOLO_IMAGE_WIDTH: int = _MY_CFG.model.yolo_image_input_width
YOLO_IMAGE_HEIGHT: int = _MY_CFG.model.yolo_image_input_height

# LOGGING LEVELS
def _get_logging_lvl_from_env(env_var_name: str) -> int:
    env_var: str | int = getenv(env_var_name, logging.DEBUG)
    try:
        return int(env_var)
    except ValueError:
        return MyLogger.get_lvl_int(str(env_var))
TIMING_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.TIMING_LOGGING_LVL.value)
FILESYSTEM_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.FILESYSTEM_LOGGING_LVL.value)
SCRIPTS_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.SCRIPTS_LOGGING_LVL.value)
IMAGE_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.IMAGE_LOGGING_LVL.value)
MODEL_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.MODEL_LOGGING_LVL.value)
CAMERA_LOGGING_LVL: int = _get_logging_lvl_from_env(EnvVars.CAMERA_LOGGING_LVL.value)
cv2.setLogLevel(0) #Default 3
logging.getLogger('ultralytics').setLevel(logging.WARNING)
