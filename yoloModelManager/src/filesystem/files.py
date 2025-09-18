from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
import yaml
from psutil import disk_usage
from pyUtils import Styles

from ..utils.config import IMAGES_PATH, my_logger
from ..utils.data_types import DatasetMetadataDict

ALLOWED_IMAGES_EXTENSIONS: set[str] = {
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.tiff'
}

def create_dataset_medatada_yaml(
    dir_path: Optional[Path],
    data: DatasetMetadataDict
) -> None:
    if dir_path is None:
        dir_path = IMAGES_PATH
    file_path: Path = dir_path / 'metadata.yaml'
    data['date'] = datetime.now(timezone.utc)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, sort_keys= False)
    my_logger.debug(f'{file_path.name} created on {file_path.parent}.', Styles.SUCCEED)

def save_image(
    image: np.ndarray,
    dir_path: Optional[Path] = None
) -> Path:
    if dir_path is None:
        dir_path = IMAGES_PATH
    image_name: str = f'{uuid4()}.png'
    image_path: Path = dir_path / image_name
    dir_path.mkdir(
        parents= True,
        exist_ok= True
    )
    if disk_usage('/').percent < 100: #TODO: cahnge to 80
        if cv2.imwrite(str(image_path), image):
            my_logger.debug(f'New image saved to "{image_path}"', Styles.SUCCEED)
        else:
            msg: str = f'Failed to save image to "{image_path.parent}". Check if directory exists.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
    else:
        my_logger.warning(f'Can\'t save image to "{image_path.parent}". Disk is full.')
    return image_path
