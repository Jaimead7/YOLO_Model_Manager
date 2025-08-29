from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from pyUtils import MyLogger, Styles

from ..utils import IMAGES_PATH, MODEL_LOGGING_LVL, ModelMetadataDict

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= MODEL_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= True
)


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
