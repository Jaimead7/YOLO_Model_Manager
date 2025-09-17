#---------- LOAD ENV VARIABLES FIRST ----------#
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(
    Path(__file__).parent / 'dist' / '.env',
    override= False
)
#---------------------------------------------#

from pyUtils import Styles

from .src.cameras import *
from .src.filesystem import *
from .src.image import *
from .src.model import *
from .src.utils import *

my_logger.debug(f'Package loaded: yoloModelManager', Styles.SUCCEED)
