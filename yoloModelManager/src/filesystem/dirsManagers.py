from pathlib import Path
from random import shuffle
from typing import Any

import yaml
from model.data import ModelTasks, ModelTrainingDataDict
from utils.config import (ALLOWED_IMAGES_EXTENSIONS, DATASETS_PATH,
                          FILESYSTEM_LOGGING_LVL, IMAGES_PATH)
from pyUtils import MyLogger, Styles

from .dirs import check_dir_path, unzip_dir
from .files import copy_files

my_logger = MyLogger(f'{__name__}', FILESYSTEM_LOGGING_LVL)


class DatasetDirManager:
    def __init__(self, path: Path, create: bool = True) -> None:
        self.create: bool = create
        self.path = path

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: Any) -> None:
        self._path: Path = check_dir_path(value, self.create)
        self._images_path: Path = check_dir_path(self._path / 'images', self.create)
        self._labels_path: Path = check_dir_path(self._path / 'labels', self.create)

    @property
    def images_path(self) -> Path:
        return self._images_path

    @property
    def labels_path(self) -> Path:
        return self._labels_path

    @property
    def metadata_path(self) -> Path:
        return self.path / 'metadata.yaml'

    def get_images_list(self) -> list[Path]:
        return [path for path in self.images_path.rglob('*')]

    def get_labels_list(self) -> list[Path]:
        return [path for path in self.labels_path.rglob('*')]

    def addData(
        self,
        images: list[Path],
        labels: list[Path]
    ) -> None:
        copy_files(images, self.images_path)
        labels_new_names: list[str]= [
            image.stem + '.txt'
            for image in images
            for label in labels
            if label.stem.endswith(image.stem)
        ]
        copy_files(labels, self.labels_path, labels_new_names)
        my_logger.debugLog(f'Data added to "{self.path}".', Styles.SUCCEED)

    def addImages(self, images_path: Path) -> None:
        if not images_path.is_absolute():
            images_path = IMAGES_PATH / images_path
        if not images_path.is_dir():
            msg: str = f'"{images_path}" does not exists.'
            my_logger.errorLog(msg)
            raise NotADirectoryError(msg)
        images: list[Path] = [
            file
            for file in images_path.iterdir()
            if file.is_file() and file.suffix.lower() in ALLOWED_IMAGES_EXTENSIONS
        ]
        copy_files(images, self.images_path)
        if (images_path / 'metadata.yaml').is_file():
            copy_files([images_path / 'metadata.yaml'], self.path)
        my_logger.debugLog(f'Images copied from "{images_path}" to "{self.images_path}".', Styles.SUCCEED)


class TrainingDatasetDirManager:
    def __init__(
        self,
        source_dataset_dir: str | Path,
    ) -> None:
        self.source_dataset_dir = Path(source_dataset_dir)

    @property
    def source_dataset_dir(self) -> DatasetDirManager:
        return self._source_dataset_dir

    @source_dataset_dir.setter
    def source_dataset_dir(self, path: Path) -> None:
        if not path.is_absolute():
            path = DATASETS_PATH / path
        path = unzip_dir(path)
        self._source_dataset_dir = DatasetDirManager(path, create= False)
        self.dataset_name = path.stem + '_split'
        self.set_paths()

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(f'"{self.__class__.__name__}.dataset_name" should be an str.')
        self._dataset_name: str = value
        self.set_paths()

    @property
    def data_yaml_file_path(self) -> Path:
        return self.path / 'data.yaml'

    @property
    def metadata_yaml_file_path(self) -> Path:
        return self.path / 'metadata.yaml'

    def set_paths(self) -> None:
        try:
            self.path: Path = self.source_dataset_dir.path.parent / self.dataset_name
            self.train_dir: DatasetDirManager = DatasetDirManager(self.path / 'train')
            self.validation_dir: DatasetDirManager = DatasetDirManager(self.path / 'validation')
            self.test_dir: DatasetDirManager = DatasetDirManager(self.path / 'test')
        except AttributeError:
            my_logger.warningLog('Can\'t set paths. No "source_dataset_dir" or "dataset_name".')

    def split(
        self,
        validation: float = 0.2,
        test: float = 0.1
    ) -> None:
        images: list[Path] = self.source_dataset_dir.get_images_list()
        labels: list[Path] = self.source_dataset_dir.get_labels_list()
        n_val: int = int(len(images) * validation)
        n_test: int = int(len(images) * test)
        shuffle(images)
        shuffle_labels: list[Path] = [
            label
            for image in images
            for label in labels
            if label.stem.endswith(image.stem)
        ]
        images_lists: list[list[Path]] = [
            images[n_val+n_test:], #Train
            images[:n_val], #Val
            images[n_val:n_val+n_test], #Test
        ]
        labels_lists: list[list[Path]] = [
            shuffle_labels[n_val+n_test:], #Train
            shuffle_labels[:n_val], #Val
            shuffle_labels[n_val:n_val+n_test], #Test
        ]
        self.train_dir.addData(images_lists[0], labels_lists[0])
        self.validation_dir.addData(images_lists[1], labels_lists[1])
        self.test_dir.addData(images_lists[2], labels_lists[2])
        if self.source_dataset_dir.metadata_path.is_file():
            copy_files([self.source_dataset_dir.metadata_path], self.path)
        my_logger.debugLog(f'{self.source_dataset_dir.path.name} splited into {self.path.name}.', Styles.SUCCEED)

    def create_yaml_data_file(self) -> None:
        classes_path: Path = self.source_dataset_dir.path / 'classes.txt'
        with open(classes_path, 'r') as f:
            classes: dict[int, str] = {
                i: line.strip()
                for i, line in enumerate(f.readlines())
                if len(line.strip()) > 0
            }
        n_classes: int = len(classes)
        data: ModelTrainingDataDict = {
            'path': str(self.path),
            'task': ModelTasks.DETECT,
            'train': 'train/images',
            'val': 'validation/images',
            'test': 'test/images',
            'nc': n_classes,
            'name': classes
        }
        with open(self.data_yaml_file_path, 'w') as f:
            yaml.dump(data, f, sort_keys= False)
        my_logger.debugLog(f'{self.data_yaml_file_path.name} created on {self.data_yaml_file_path.parent}.', Styles.SUCCEED)
