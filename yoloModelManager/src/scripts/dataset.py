import logging
from pathlib import Path

import click

from ..filesystem import TrainingDatasetDirManager
from ..utils.config import (LOGGING_LVL, my_logger, save_yolo_manager_logs,
                            set_yolo_manager_logging_level,
                            set_yolo_manager_logs_path)


@click.command()
@click.option(
    '--data-source',
    '-d',
    'data_source',
    type= click.Path(
        readable= True,
        writable= True,
        path_type= Path
    ),
    required= True,
    help= f'Path to the exported dataset from label-studio. Train/Validation/Test directories will be created on parent with name <data_source>_split.'
)
@click.option(
    '--images',
    '-i',
    'images_source',
    type= click.Path(
        readable= True,
        path_type= Path
    ),
    help= f'Path to the directory with the images.'
)
@click.option(
    '--validation',
    '-v',
    'validation',
    type= click.FloatRange(
        min= 0,
        max= 1,
        min_open= True,
        max_open= True
    ),
    default= 0.2,
    help= '% of the images for validation. Default to 0.2.'
)
@click.option(
    '--test',
    '-t',
    'test',
    type= click.FloatRange(
        min= 0,
        max= 1,
        min_open= True,
        max_open= True
    ),
    default= 0.1,
    help= '% of the images for test. Default to 0.1.'
)
def split_dataset(
    data_source: Path,
    images_source: Path,
    validation: float = 0.2,
    test: float = 0.1
) -> None:
    set_yolo_manager_logging_level(logging.WARNING)
    set_yolo_manager_logs_path('yoloModelManager.log')
    set_yolo_manager_logging_level(LOGGING_LVL)
    save_yolo_manager_logs(True)
    my_logger.debug(f'Executed: split-dataset -d {data_source} -i {images_source} -v {validation} -t {test}')
    if validation + test > 0.5:
        msg: str = 'Validation + test ratio should be lower than 50% of the dataset.'
        my_logger.error(f'AttributeError: {msg}')
        raise AttributeError(msg)
    dirManager: TrainingDatasetDirManager = TrainingDatasetDirManager(
        source_dataset_dir= data_source
    )
    if images_source is not None:
        dirManager.source_dataset_dir.add_images(images_source)
    dirManager.split(validation= validation, test= test)
