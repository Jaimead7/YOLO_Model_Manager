from pathlib import Path
from shutil import copy2
from typing import Optional

from utils.config import FILESYSTEM_LOGGING_LVL
from pyUtils import MyLogger, Styles

my_logger = MyLogger(f'{__name__}', FILESYSTEM_LOGGING_LVL)

def copy_files(
    files_list: list[Path],
    destiny_dir: Path,
    new_names: Optional[list[str]] = None
) -> None:
    if not destiny_dir.is_dir():
        msg: str = f'"{destiny_dir}" does not exists.'
        my_logger.error(f'NotADirectoryError: {msg}')
        raise NotADirectoryError(msg)
    if new_names is None:
        new_names = [file.name for file in files_list]
    destiny_files: list[Path] = [destiny_dir / new_name for new_name in new_names]
    for source, destiny in zip(files_list, destiny_files):
        if source.is_file():
            copy2(source, destiny)
            my_logger.debug(f'"{source.name}" copied to "{destiny}".', Styles.SUCCEED)
        else:
            my_logger.warning(f'"{source}" won\'t be copied. File doesn\'t exists.')
