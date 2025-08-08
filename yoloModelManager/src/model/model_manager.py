from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from image.image_processing import ImageProcessing
from ultralytics import YOLO
from utils.config import MODEL_LOGGING_LVL, MODELS_PATH
from pyUtils import MyLogger, Styles

from .data import Model_Metadata_Dict

my_logger = MyLogger(f'{__name__}', MODEL_LOGGING_LVL)


class Model_Manager:
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f'"{self.__class__.__name__}.name" should be a str.')
        path: Path = MODELS_PATH / value
        pt_model_path: Path = path / (value + '.pt')
        metadata_path: Path = path / 'metadata.yaml'
        if not path.is_dir():
            raise NotADirectoryError(f'"{path}" does not exists.')
        if not pt_model_path.is_file() or not metadata_path.is_file():
            raise FileExistsError(f'{path} structure error. The directory must contain "{value + ".pt"}" and "metadata.yaml".')
        self._name: str = value
        self._path: Path = path
        self._pt_model_path: Path = pt_model_path
        self._metadata_path: Path = metadata_path
        self._exportModel2NCNN()
        self._loadModel()

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
    def metadata(self) -> Model_Metadata_Dict:
        with open(self._metadata_path, 'r') as f:
            metadata: Model_Metadata_Dict = yaml.safe_load(f) #TODO: validate file
        return metadata

    def _exportModel2NCNN(self) -> None:
        self.ncnn_model_path: Path = self.pt_model_path.with_name(self.pt_model_path.stem + '_ncnn_model')
        if self._isValidNCNN(self.ncnn_model_path):
            my_logger.warningLog(f'Model "{self.pt_model_path.stem}" not exported. NCNN model already exists.')
            return
        model = YOLO(self.pt_model_path)
        model.export(format= 'ncnn') #FIXME: verbose= False
        self.pt_model_path.with_suffix('.torchscript').unlink()
        (self.ncnn_model_path / 'model_ncnn.py').unlink()
        my_logger.debugLog(f'Model "{self.pt_model_path.stem}" exported to NCNN.', Styles.SUCCEED)

    def _isValidNCNN(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        filesList: list[str] = [
            file.name
            for file in path.iterdir()
            if file.is_file()
        ]
        if not all(
            file in filesList
            for file in (
                'metadata.yaml',
                'model.ncnn.bin',
                'model.ncnn.param'
            )
        ):
            return False
        return True

    def _loadModel(self) -> None:
        self.model = YOLO(
            self.ncnn_model_path,
            task= 'detect'
        )
        my_logger.debugLog(f'Model "{self.ncnn_model_path.stem}" loaded.', Styles.SUCCEED)

    def processFrame(self, frame: np.ndarray) -> np.ndarray:
        frames: list[np.ndarray] = [frame]
        for filter in self.filters:
            frames.append(filter(frames[-1]))
        results: list = self.model(frames[-1])
        frames.append(results[0].plot())
        self.last_input: np.ndarray = frame
        self.last_processed: np.ndarray = frames[-2]
        self.last_result: np.ndarray = frames[-1]
        #TODO: change to return results
        return ImageProcessing.get_images_grid([self.last_input, self.last_result])
