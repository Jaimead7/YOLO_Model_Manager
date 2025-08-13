from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, TypedDict

import yaml
from pyUtils import MyLogger, Styles

from ..utils import IMAGES_PATH, MODEL_LOGGING_LVL

my_logger = MyLogger(f'{__name__}', MODEL_LOGGING_LVL)


class ModelTasks(Enum):
    DETECT = 'detect'
    SEGMENT = 'segment'
    CLASSIFY = 'classify'
    POSE = 'pose'
    

class ModelTrainingDataDict(TypedDict):
    path: str
    task: ModelTasks
    train: str
    val: str
    test: str
    nc: int
    name: dict[int, str]


class ModelMetadataDict(TypedDict):
    date: datetime
    camera_width: int
    camera_height: int
    filters: list[str]


def create_model_medatada_yaml(
    dir_path: Optional[Path],
    width: int,
    height: int,
    filters: list[str]
) -> None:
    if dir_path is None:
        dir_path = IMAGES_PATH
    file_path: Path = dir_path / 'metadata.yaml'
    data: ModelMetadataDict = {
        'date': datetime.now(timezone.utc),
        'camera_width': width,
        'camera_height': height,
        'filters': filters
    }
    with open(file_path, 'w') as f:
        yaml.dump(data, f, sort_keys= False)
    my_logger.debug(f'{file_path.name} created on {file_path.parent}.', Styles.SUCCEED)
