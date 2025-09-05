from pathlib import Path
from random import shuffle
from typing import Any, Optional

import yaml
from pyUtils import MyLogger, Styles

from ..utils import (DATASETS_PATH, FILESYSTEM_LOGGING_LVL, IMAGES_PATH,
                     DatasetDataDict, DatasetMetadataDict, ModelTasks)
from .dirs import check_dir_path, unzip_dir
from .files import ALLOWED_IMAGES_EXTENSIONS, copy_files

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= FILESYSTEM_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)


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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  • path= "{self.path}"",\n'
                f'  • images_path= "{self.images_path}",\n'
                f'  • labels_path= "{self.labels_path}",\n'
                f'  • metadata_path= "{self.metadata_path}"\n)')

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}("{self.path}")')

    def get_images_list(self) -> list[Path]:
        return [path for path in self.images_path.rglob('*')]

    def get_labels_list(self) -> list[Path]:
        return [path for path in self.labels_path.rglob('*')]

    def add_data(
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
        my_logger.debug(f'Data added to "{self.path}".', Styles.SUCCEED)

    def add_images(self, images_path: Path) -> None:
        if not images_path.is_absolute():
            images_path = IMAGES_PATH / images_path
        if not images_path.is_dir():
            msg: str = f'"{images_path}" does not exists.'
            my_logger.error(f'NotADirectoryError: {msg}')
            raise NotADirectoryError(msg)
        images: list[Path] = [
            file
            for file in images_path.iterdir()
            if file.is_file() and file.suffix.lower() in ALLOWED_IMAGES_EXTENSIONS
        ]
        copy_files(images, self.images_path)
        if (images_path / 'metadata.yaml').is_file():
            copy_files([images_path / 'metadata.yaml'], self.path)
        my_logger.debug(f'Images copied from "{images_path}" to "{self.images_path}".', Styles.SUCCEED)

    def get_n_images(self) -> int:
        return len(self.get_images_list())


class TrainingDatasetDirManager:
    def __init__(
        self,
        dataset_dir: Optional[str | Path] = None,
        source_dataset_dir: Optional[str | Path] = None
    ) -> None:
        if dataset_dir is not None:
            self.path = Path(dataset_dir)
            self.set_paths()
            return
        if source_dataset_dir is not None:
            self.source_dataset_dir = Path(source_dataset_dir)
            name: str = self.source_dataset_dir.path.stem + '_split'
            self.path = self.source_dataset_dir.path.parent / name
            self.create_paths()
            return
        msg: str = f'"dataset_dir" or "source_dataset_dir" needs to be defined.'
        my_logger.error(f'AttributeError: {msg}')
        raise AttributeError(msg)

    @property
    def source_dataset_dir(self) -> DatasetDirManager:
        if self._source_dataset_dir is not None:
            return self._source_dataset_dir
        msg: str = f'"source_dataset_dir" is not defined.'
        my_logger.error(f'AttributeError: {msg}')
        raise AttributeError(msg)

    @source_dataset_dir.setter
    def source_dataset_dir(self, path: Path) -> None:
        if not path.is_absolute():
            path = DATASETS_PATH / path
        path = unzip_dir(path)
        self._source_dataset_dir: Optional[DatasetDirManager] = DatasetDirManager(
            path,
            create= False
        )

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, path: Path) -> None:
        if not path.is_absolute():
            path = DATASETS_PATH / path
        self._path: Path = path

    @property
    def dataset_name(self) -> str:
        return self.path.stem

    @property
    def data_yaml_file_path(self) -> Path:
        return self.path / 'data.yaml'

    @property
    def data(self) -> DatasetDataDict:
        with open(self.data_yaml_file_path, 'r') as f:
            data: DatasetDataDict = yaml.safe_load(f) #TODO: validate file
        return data

    @property
    def metadata_yaml_file_path(self) -> Path:
        return self.path / 'metadata.yaml'

    @property
    def metadata(self) -> DatasetMetadataDict:
        with open(self.metadata_yaml_file_path, 'r') as f:
            metadata: DatasetMetadataDict = yaml.safe_load(f) #TODO: validate file
        return metadata

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  • path= "{self.path}",\n'
                f'  • train_dir= {self.train_dir},\n'
                f'  • validation_dir= {self.validation_dir},\n'
                f'  • test_dir= {self.test_dir}\n)')

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}("{self.path}")')

    def set_paths(self) -> None:
        try:
            self.train_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'train',
                create= False
            )
            self.validation_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'validation',
                create= False
            )
            self.test_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'test',
                create= False
            )
        except AttributeError:
            my_logger.warning('Can\'t set paths. No "path" is defined.')

    def create_paths(self) -> None:
        try:
            self.train_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'train',
                create= True
            )
            self.validation_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'validation',
                create= True
            )
            self.test_dir: DatasetDirManager = DatasetDirManager(
                self.path / 'test',
                create= True
            )
        except AttributeError:
            my_logger.warning('Can\'t set paths. No "path" is defined.')

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
        self.train_dir.add_data(images_lists[0], labels_lists[0])
        self.validation_dir.add_data(images_lists[1], labels_lists[1])
        self.test_dir.add_data(images_lists[2], labels_lists[2])
        if self.source_dataset_dir.metadata_path.is_file():
            copy_files([self.source_dataset_dir.metadata_path], self.path)
        self.create_yaml_data_file()
        my_logger.debug(
            f'{self.source_dataset_dir.path.name} splited into {self.path.name}.',
            Styles.SUCCEED
        )

    def create_yaml_data_file(self) -> None:
        classes_path: Path = self.source_dataset_dir.path / 'classes.txt'
        with open(classes_path, 'r') as f:
            classes: dict[int, str] = {
                i: line.strip()
                for i, line in enumerate(f.readlines())
                if len(line.strip()) > 0
            }
        n_classes: int = len(classes)
        data: DatasetDataDict = {
            'path': str(self.path),
            'task': str(ModelTasks.DETECT.value),
            'train': 'train/images',
            'val': 'validation/images',
            'test': 'test/images',
            'nc': n_classes,
            'name': classes,
        }
        with open(self.data_yaml_file_path, 'w') as f:
            yaml.dump(data, f, sort_keys= False)
        my_logger.debug(
            f'{self.data_yaml_file_path.name} created on {self.data_yaml_file_path.parent}.',
            Styles.SUCCEED
        )

    def get_n_train(self) -> int:
        return self.train_dir.get_n_images()

    def get_n_val(self) -> int:
        return self.validation_dir.get_n_images()

    def get_n_test(self) -> int:
        return self.test_dir.get_n_images()
