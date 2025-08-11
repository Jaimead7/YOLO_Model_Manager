from pathlib import Path
from typing import Callable, Optional

import click
from cameras.camera_manager import CameraManager, camera_manager_factory
from image.image_processing import ImageProcessing
from utils.config import SCRIPTS_LOGGING_LVL
from pyUtils import MyLogger

my_logger = MyLogger(f'{__name__}', SCRIPTS_LOGGING_LVL)


@click.command()
@click.option(
    '--camera',
    '-c',
    'camera',
    type= click.IntRange(0),
    help= 'Id of the camera for opencv. Defaults to None for select.'
)
@click.option(
    '--show-filter',
    '-f',
    'show_filters_in',
    multiple= True,
    type= click.Choice(
        ImageProcessing.FILTERS.keys(),
        case_sensitive= False
    ),
    help= f'Processing filter for showing images. Valid options: {ImageProcessing.FILTERS.keys()}'
)
@click.option(
    '--save-filter',
    '-s',
    'save_filters_in',
    multiple= True,
    type= click.Choice(
        ImageProcessing.FILTERS.keys(),
        case_sensitive= False
    ),
    help= f'Processing filter for saving images. Valid options: {ImageProcessing.FILTERS.keys()}'
)
@click.option(
    '--save-path',
    '-p',
    'save_path',
    type= click.Path(
        dir_okay= True,
        writable= True,
        path_type= Path
    ),
    help= 'Path to save the file. If None try to import from environment variable "IMAGES_SAVE_PATH". Else set to app/images.'
)
def image_adquisition(
    camera: int,
    show_filters_in: list[str] = [],
    save_filters_in: Optional[str] = None,
    save_path: Optional[Path] = None
) -> None:
    my_logger.debug(f'Executed: image-adquisition -c {camera} -f {show_filters_in} -s {save_filters_in} -p {save_path}')
    show_filters: list[Callable] = [ImageProcessing.FILTERS[filter] for filter in show_filters_in]
    if save_filters_in is None:
        save_filters: Optional[list[Callable]] = None
    else:
        save_filters: Optional[list[Callable]] = [ImageProcessing.FILTERS[filter] for filter in save_filters_in]
    camera_manager: CameraManager = camera_manager_factory(camera)
    camera_manager.video_stream(
        show_filters= show_filters,
        save_filters= save_filters,
        save_dir_path= save_path
    )
