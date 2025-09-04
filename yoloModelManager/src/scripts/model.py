from pathlib import Path
from typing import Optional

import click
from pyUtils import MyLogger

from ..cameras import CameraManager, camera_manager_factory
from ..filesystem import TrainingDatasetDirManager
from ..model import ModelManager
from ..utils import SCRIPTS_LOGGING_LVL, YOLO_IMAGE_WIDTH

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= SCRIPTS_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)


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
    model: ModelManager = ModelManager(model_name)
    camera_manager: CameraManager = camera_manager_factory(camera)
    camera_manager.save_dir_path = save_path
    camera_manager.load_params_from_model(model)
    camera_manager.keys_callbacks = {
        32: (camera_manager.save_last_frame, {})
    }
    camera_manager.video_stream()


@click.command()
@click.option(
    '--name',
    '-n',
    'name',
    type= click.STRING,
    required= True,
    help= 'Name of the model to create.'
)
@click.option(
    '--base-model',
    '-m',
    'base_model',
    type= click.STRING,
    required= True,
    help= 'Name of the model to be used as base.'
)
@click.option(
    '--dataset',
    '-d',
    'dataset',
    type= click.Path(
        readable= True,
        writable= True,
        path_type= Path
    ),
    required= True,
    help= f'Path to the dataset.'
)
@click.option(
    '--epochs',
    '-e',
    'epochs',
    type= click.IntRange(
        min= 0,
        min_open= True,
    ),
    default= 60,
    help= 'Epoch of the training process.'
)
def train_model(
    name: str,
    base_model: str,
    dataset: Path,
    epochs: int,
) -> None:
    my_logger.debug(f'Executed: train-model -n {name} -m {base_model} -d {dataset} -e {epochs}')
    dataset_dir: TrainingDatasetDirManager = TrainingDatasetDirManager(
        dataset_dir= dataset
    )
    base_model_manager: ModelManager = ModelManager(base_model)
    base_model_manager.train(
        dataset= dataset_dir,
        new_name= name,
        epochs= epochs
    )
