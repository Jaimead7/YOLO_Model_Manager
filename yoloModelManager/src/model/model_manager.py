import logging
from datetime import datetime, timezone
from os import getcwd
from pathlib import Path
from shutil import copy2
from typing import Any, Callable

import numpy as np
import yaml
from pyUtils import MyLogger, Styles
from ultralytics import YOLO

from ..filesystem import TrainingDatasetDirManager
from ..image import ImageProcessing
from ..utils import (MODEL_LOGGING_LVL, MODELS_PATH, ULTRALYTICS_LOGGING_LVL,
                     ModelMetadataDict)
from .results import ResultTracker

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= MODEL_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)


class ModelManager:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        if not isinstance(value, str):
            msg: str = f'"{self.__class__.__name__}.name" should be a str.'
            my_logger.error(f'TypeError: {msg}')
            raise TypeError(msg)
        path: Path = MODELS_PATH / value
        pt_model_path: Path = path / (value + '.pt')
        metadata_path: Path = path / 'metadata.yaml'
        if not path.is_dir():
            msg: str = f'"{path}" does not exists.'
            my_logger.error(f'NotADirectoryError: {msg}')
            raise NotADirectoryError(msg)
        if not pt_model_path.is_file() or not metadata_path.is_file():
            msg: str = f'{path} structure error. The directory must contain "{value + ".pt"}" and "metadata.yaml".'
            my_logger.error(f'FileExistsError: {msg}')
            raise FileExistsError(msg)
        self._name: str = value
        self._path: Path = path
        self._pt_model_path: Path = pt_model_path
        self._metadata_path: Path = metadata_path
        self._export_model_2_ncnn()
        self._load_model()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def pt_model_path(self) -> Path:
        return self._pt_model_path

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    @property
    def metadata(self) -> ModelMetadataDict:
        with open(self._metadata_path, 'r') as f:
            metadata: ModelMetadataDict = yaml.safe_load(f)
        return metadata

    @property
    def date(self) -> datetime:
        return self.metadata['date']

    @property
    def camera_width(self) -> int:
        return self.metadata['camera_width']

    @property
    def camera_height(self) -> int:
        return self.metadata['camera_height']

    @property
    def filters(self) -> list[Callable[..., Any]]:
        return [
            ImageProcessing.FILTERS[filter]
            for filter in self.metadata['filters']
        ]

    @property
    def camera_brightness(self) -> float:
        return self.metadata['brightness']

    @property
    def camera_contrast(self) -> float:
        return self.metadata['contrast']

    @property
    def camera_saturation(self) -> float:
        return self.metadata['saturation']

    @property
    def camera_exposure(self) -> float:
        return self.metadata['exposure']

    @property
    def camera_wb(self) -> float:
        return self.metadata['wb']

    @property
    def train_images(self) -> int:
        return self.metadata['train_images']

    @property
    def val_images(self) -> int:
        return self.metadata['val_images']

    @property
    def test_images(self) -> int:
        return self.metadata['test_images']

    @property
    def task(self) -> str:
        return self.metadata['task']

    @property
    def object_classes(self) -> dict[int, str]:
        return self.metadata['name']

    def _export_model_2_ncnn(self) -> None:
        self.ncnn_model_path: Path = self.pt_model_path.with_name(self.pt_model_path.stem + '_ncnn_model')
        if self._is_valid_ncnn(self.ncnn_model_path):
            my_logger.warning(f'Model "{self.pt_model_path.stem}" not exported. NCNN model already exists.')
            return
        model = YOLO(self.pt_model_path)
        model.export(format= 'ncnn') #FIXME: verbose= False
        self.pt_model_path.with_suffix('.torchscript').unlink()
        (self.ncnn_model_path / 'model_ncnn.py').unlink()
        my_logger.debug(f'Model "{self.pt_model_path.stem}" exported to NCNN.', Styles.SUCCEED)

    def _is_valid_ncnn(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        files_list: list[str] = [
            file.name
            for file in path.iterdir()
            if file.is_file()
        ]
        if not all(
            file in files_list
            for file in (
                'metadata.yaml',
                'model.ncnn.bin',
                'model.ncnn.param'
            )
        ):
            return False
        return True

    def _load_model(self) -> None:
        self.model = YOLO(
            self.ncnn_model_path,
            task= 'detect'
        )
        self.result_tracker: ResultTracker = ResultTracker()
        my_logger.debug(f'Model "{self.ncnn_model_path.stem}" loaded.', Styles.SUCCEED)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frames: list[np.ndarray] = [frame]
        for filter in self.filters:
            frames.append(filter(frames[-1]))
        self.result_tracker.add_new_result(self.model(frames[-1])[0])
        frames.append(self.result_tracker.plot())
        self.last_input_img: np.ndarray = frame
        self.last_processed_img: np.ndarray = frames[-2]
        self.last_result_img: np.ndarray = frames[-1]
        return frames[-1]

    def get_last_result_image(self, source: bool = True) -> np.ndarray:
        if source:
            return ImageProcessing.get_images_grid(
                [self.last_input_img, self.last_result_img]
            )
        return self.last_result_img

    def train(
        self,
        dataset: TrainingDatasetDirManager,
        new_name: str,
        epochs: int = 60,
    ) -> None:
        if any([
            dataset.metadata['camera_width'] != self.metadata['camera_width'],
            dataset.metadata['camera_height'] != self.metadata['camera_height'],
            dataset.metadata['filters'] != self.metadata['filters']
        ]):
            msg: str = f'Base model and dataset must have the same image size and filters.'
            my_logger.error(f'ValueError: {msg}')
            raise ValueError(msg)
        new_model_path: Path = MODELS_PATH / new_name
        if new_model_path.is_dir():
            my_logger.warning(f'The model already exists. {new_name}.pt won\'t be overwritten. metadata.yaml will be overwrite.')
        model = YOLO(self.pt_model_path)
        cwd: Path = Path(getcwd())
        start_time: datetime = datetime.now(timezone.utc)
        logging.getLogger('ultralytics').setLevel(my_logger._logger.level)
        model.train(
            data= dataset.data_yaml_file_path,
            epochs= epochs,
            imgsz= dataset.metadata['camera_width'],
            project= new_model_path,
            batch= -1,
            verbose= True
        )
        logging.getLogger('ultralytics').setLevel(ULTRALYTICS_LOGGING_LVL)
        my_logger.debug(
            f'Training for "{new_name}" finished in {datetime.now(timezone.utc) - start_time}.',
            Styles.SUCCEED
        )
        data: ModelMetadataDict = {
            **dataset.metadata,
            'train_images': dataset.get_n_train(),
            'val_images': dataset.get_n_val(),
            'test_images': dataset.get_n_test(),
            'task': dataset.data['task'],
            'name': dataset.data['name']
        }
        with open((new_model_path / 'metadata.yaml'), 'w') as f:
            yaml.dump(data, f, sort_keys= False)
        copy2(
            (new_model_path / 'train' / 'weights' / 'best.pt'),
            (new_model_path / f'{new_name}.pt')
        )
        try:
            (cwd / 'yolo11n.pt').unlink()
        except FileNotFoundError:
            pass
