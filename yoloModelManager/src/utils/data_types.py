from datetime import datetime
from enum import Enum
from typing import TypedDict


class ModelTasks(Enum):
    DETECT = 'detect'
    SEGMENT = 'segment'
    CLASSIFY = 'classify'
    POSE = 'pose'


class ModelTrainingDataDict(TypedDict):
    path: str
    task: str
    train: str
    val: str
    test: str
    nc: int
    name: dict[int, str]


class DatasetMetadataDict(TypedDict):
    date: datetime
    camera_width: int
    camera_height: int
    filters: list[str]
    brightness: float
    contrast: float
    saturation: float
    auto_exposure: float
    exposure: float
    auto_wb: float
    wb: float


class ModelMetadataDict(DatasetMetadataDict):
    train_images: int
    val_images: int
    test_images: int
    task: str
    name: dict[int, str]
