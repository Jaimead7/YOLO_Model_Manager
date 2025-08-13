from pathlib import Path
from typing import Optional

import click
from pyUtils import MyLogger

from ..cameras import CameraManager, camera_manager_factory
from ..model import ModelManager
from ..utils import SCRIPTS_LOGGING_LVL

my_logger = MyLogger(f'{__name__}', SCRIPTS_LOGGING_LVL)


@click.command()
@click.option(
    '--model',
    '-m',
    'model_name',
    type= click.STRING,
    required= True,
    help= 'Name of the model to be used.'
)
@click.option(
    '--camera',
    '-c',
    'camera',
    type= click.IntRange(0),
    help= 'Id of the camera for opencv. Defaults to None for select.'
)
@click.option(
    '--save-path',
    '-p',
    'save_path',
    type= click.Path(
        exists= True,
        dir_okay= True,
        writable= True,
        path_type= Path
    ),
    help= 'Path to save the file. If None try to import from environment variable "IMAGES_SAVE_PATH". Else set to app/images.'
)
def test_model(
    model_name: str,
    camera: int,
    save_path: Optional[Path] = None
) -> None:
    my_logger.debug(f'Executed: test-model -m {model_name} -c {camera} -p {save_path}')
    camera_manager: CameraManager = camera_manager_factory(camera)
    model: ModelManager = ModelManager(model_name)
    camera_manager.model_stream(
        model= model,
        save_dir_path= save_path
    )


@click.command()
@click.option(
    '--model',
    '-m',
    'base_model',
    type= click.STRING,
    required= True,
    help= 'Name of the model to be used.'
)
def train_model(
    base_model: Path,
    dataset: str,
    epochs: int,
    imgsz: int
) -> None:
    my_logger.debug(f'Executed: train-model -m {base_model} -d {dataset} -e {epochs} -s {imgsz}')
    ... #TODO
