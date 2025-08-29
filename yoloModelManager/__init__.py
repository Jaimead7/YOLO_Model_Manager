from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    Path(__file__).parent / 'dist' / '.env',
    override= False
)

import pyUtils

pyUtils.set_pyutils_logs_path('yoloModelManager.log')
pyUtils.save_pyutils_logs(True)

from .src.cameras import *
from .src.filesystem import *
from .src.image import *
from .src.model import *
from .src.utils import *
