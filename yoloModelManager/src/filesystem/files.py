from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Optional
from uuid import uuid4

import cv2
import numpy as np
import yaml
from pyUtils import MyLogger, Styles

from ..utils.config import FILESYSTEM_LOGGING_LVL, IMAGES_PATH
from ..utils.data_types import DatasetMetadataDict

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= FILESYSTEM_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)

ALLOWED_IMAGES_EXTENSIONS: set[str] = {
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.tiff'
}

def copy_files(
    files_list: list[Path],
    destiny_dir: Path,
    new_names: Optional[list[str]] = None
) -> None:
    if not destiny_dir.is_dir():
        msg: str = f'"{destiny_dir}" does not exists.'
        my_logger.error(f'NotADirectoryError: {msg}')
        raise NotADirectoryError(msg)
    if new_names is None:
        new_names = [file.name for file in files_list]
    destiny_files: list[Path] = [destiny_dir / new_name for new_name in new_names]
    for source, destiny in zip(files_list, destiny_files):
        if source.is_file():
            copy2(source, destiny)
            my_logger.debug(f'"{source.name}" copied to "{destiny}".', Styles.SUCCEED)
        else:
            my_logger.warning(f'"{source}" won\'t be copied. File doesn\'t exists.')

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
    if cv2.imwrite(str(image_path), image):
        my_logger.debug(f'New image saved to "{image_path}"', Styles.SUCCEED)
    else:
        msg: str = f'Failed to save image to "{image_path.parent}". Check if directory exists.'
        my_logger.error(f'RuntimeError: {msg}')
        raise RuntimeError(msg)
    return image_path