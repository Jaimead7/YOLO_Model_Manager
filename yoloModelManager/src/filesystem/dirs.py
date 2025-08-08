import tarfile
import zipfile
from pathlib import Path
from typing import Any, Literal

from utils.config import FILESYSTEM_LOGGING_LVL
from pyUtils import MyLogger, Styles

my_logger = MyLogger(f'{__name__}', FILESYSTEM_LOGGING_LVL)

def unzip_dir(dir: Path) -> Path:
    if not dir.exists():
        msg: str = f'Path "{dir}" doesn\'t exists.'
        my_logger.errorLog(msg)
        raise FileExistsError(msg)
    if dir.is_dir():
        return dir
    suffixes: list[str] = dir.suffixes
    if not suffixes:
        msg: str = f'File has no extension.'
        my_logger.errorLog(msg)
        raise ValueError(msg)
    extension: str = suffixes[-1].lower()
    newPath: Path = dir.parent / dir.stem
    if extension == '.zip':
        uzip(dir, newPath)
    elif extension in ('.tar', '.gz', '.bz2', '.xz'):
        utar(dir, newPath)
    else:
        msg: str = f'File extension not supported: "{extension}".'
        my_logger.errorLog(msg)
        raise ValueError(msg)
    return newPath

def uzip(file: Path, path: Path) -> None:
    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(path)
    my_logger.debugLog(f'{file.name} extracted in {path}.', Styles.SUCCEED)

def utar(file: Path, path: Path) -> None:
    modes: dict[str, Literal['r', 'r:gz', 'r:bz2', 'r:xz']] = {
        '.tar': 'r',
        '.gz': 'r:gz',
        '.bz2': 'r:bz2',
        '.xz': 'r:xz'
    }
    mode: str = modes[file.suffixes[-1].lower()]
    with tarfile.open(file, mode) as f:
        f.extractall(path)
    my_logger.debugLog(f'{file.name} extracted in {path}.', Styles.SUCCEED)

def check_dir_path(path_in: Any, create: bool= True) -> Path:
    try:
        path: Path = Path(path_in)
    except Exception:
        msg: str = f'Unable to get pathlib.Path from "{path_in}".'
        my_logger.errorLog(msg)
        raise TypeError(msg)
    if not path.is_dir():
        if not create:
            msg: str = f'{path} does not exists.'
            my_logger.errorLog(msg)
            raise NotADirectoryError(msg)
        try:
            path.mkdir(parents= True)
            my_logger.infoLog(f'{path} created.')
        except FileExistsError:
            pass
    return path
