from pathlib import Path
from typing import Any

from ..utils.config import my_logger


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
