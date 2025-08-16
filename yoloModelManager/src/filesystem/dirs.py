import tarfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from pyUtils import MyLogger, Styles

from ..utils import FILESYSTEM_LOGGING_LVL

my_logger = MyLogger(f'{__name__}', FILESYSTEM_LOGGING_LVL)

def unzip_dir(dir: Path) -> Path:
    if not dir.exists():
        msg: str = f'Path "{dir}" doesn\'t exists.'
        my_logger.error(f'FileExistsError: {msg}')
        raise FileExistsError(msg)
    if dir.is_dir():
        return dir
    suffixes: list[str] = dir.suffixes
    if not suffixes:
        msg: str = f'File has no extension.'
        my_logger.error(f'ValueError: {msg}')
        raise ValueError(msg)
    extension: str = suffixes[-1].lower()
    new_path: Path = dir.parent / dir.stem
    if extension == '.zip':
        _uzip(dir, new_path)
    elif extension in ('.tar', '.gz', '.bz2', '.xz'):
        _utar(dir, new_path)
    else:
        msg: str = f'File extension not supported: "{extension}".'
        my_logger.error(f'ValueError: {msg}')
        raise ValueError(msg)
    return new_path

def _uzip(file: Path, path: Path) -> None:
    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(path)
    my_logger.debug(f'{file.name} extracted in {path}.', Styles.SUCCEED)

def _utar(file: Path, path: Path) -> None:
    modes: dict[str, Literal['r', 'r:gz', 'r:bz2', 'r:xz']] = {
        '.tar': 'r',
        '.gz': 'r:gz',
        '.bz2': 'r:bz2',
        '.xz': 'r:xz'
    }
    mode: str = modes[file.suffixes[-1].lower()]
    with tarfile.open(file, mode) as f:
        f.extractall(path)
    my_logger.debug(f'{file.name} extracted in {path}.', Styles.SUCCEED)

def check_dir_path(path_in: Any, create: bool= True) -> Path:
    try:
        path: Path = Path(path_in)
    except Exception:
        msg: str = f'Unable to get pathlib.Path from "{path_in}".'
        my_logger.error(f'TypeError: {msg}')
        raise TypeError(msg)
    if not path.is_dir():
        if not create:
            msg: str = f'{path} does not exists.'
            my_logger.error(f'NotADirectoryError: {msg}')
            raise NotADirectoryError(msg)
        try:
            path.mkdir(parents= True)
            my_logger.info(f'{path} created.')
        except FileExistsError:
            pass
    return path
