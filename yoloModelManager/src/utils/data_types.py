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
    task: ModelTasks
    train: str
    val: str
    test: str
    nc: int
    name: dict[int, str]


class ModelMetadataDict(TypedDict):
    date: datetime
    camera_width: int
    camera_height: int
    filters: list[str]
