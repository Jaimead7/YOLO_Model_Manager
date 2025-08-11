from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from image.image_processing import ImageProcessing
from ultralytics import YOLO
from utils.config import MODEL_LOGGING_LVL, MODELS_PATH
from pyUtils import MyLogger, Styles

from .data import ModelMetadataDict

my_logger = MyLogger(f'{__name__}', MODEL_LOGGING_LVL)


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
    def camera_width(self) -> int:
        return self.metadata['camera_width']

    @property
    def camera_height(self) -> int:
        return self.metadata['camera_height']

    @property
    def date(self) -> datetime:
        return self.metadata['date']

    @property
    def filters(self) -> list[Callable[..., Any]]:
        return [
            ImageProcessing.FILTERS[filter]
            for filter in self.metadata['filters']
        ]

    @property
    def metadata(self) -> ModelMetadataDict:
        with open(self._metadata_path, 'r') as f:
            metadata: ModelMetadataDict = yaml.safe_load(f) #TODO: validate file
        return metadata

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
        my_logger.debug(f'Model "{self.ncnn_model_path.stem}" loaded.', Styles.SUCCEED)

    def process_frame(self, frame: np.ndarray) -> list:
        frames: list[np.ndarray] = [frame]
        for filter in self.filters:
            frames.append(filter(frames[-1]))
        results: list = self.model(frames[-1])
        frames.append(results[0].plot())
        self.last_input: np.ndarray = frame
        self.last_processed: np.ndarray = frames[-2]
        self.last_result: np.ndarray = frames[-1]
        return results

    def get_last_result_image(self, source: bool = True) -> np.ndarray:
        if source:
            return ImageProcessing.get_images_grid([self.last_input, self.last_result])
        return self.last_result
