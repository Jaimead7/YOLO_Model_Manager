# Configuration
Documentation for [config.py](../../../yoloModelManager/src/utils/config.py)  
Configuration of the package.
- [EnvVars](#envvars)
- [Paths](#paths)
- [Constants](#constants)

<br>

## EnvVars
Enum of the environment variables that the package will try to read.
- `IMAGES_PATH`: Default path for images files.
- `MODELS_PATH`: Default path for models files.
- `DATASETS_PATH`: Default path for datasets files.
- `TIMING_LOGGING_LVL`: Logging level for timing subpackage.
- `FILESYSTEM_LOGGING_LVL`: Logging level for filesystem subpackage.
- `SCRIPTS_LOGGING_LVL`: Logging level for scripts subpackage.
- `IMAGE_LOGGING_LVL`: Logging level for image subpackage.
- `MODEL_LOGGING_LVL`: Logging level for model subpackage.
- `CAMERA_LOGGING_LVL`: Logging level for camera subpackage.

<br>

## Paths
Default paths of folders used in the package.  
The package try to read the variable form [`./yoloModelManager/dist/.env`](./yoloModelManager/dist/)  
Check the structure of the file in [**`example.env`**](./docs/examples/example.env).  
Environment variables can be loaded before importing anything from the package.  
#### Example:
```python
from dotenv import load_dotenv
load_dotenv('example/path/.env')
from yoloModelManager.utils import *
```
- **IMAGES_PATH**: *Path* `<package>/dist/images`
- **MODELS_PATH**: *Path* `<package>/dist/models`
- **DATASETS_PATH**: *Path* `<package>/dist/datasets`

<br>

## Constants
Constants of the package.  
Loaded from [config.toml](../../../yoloModelManager/dist/config/config.toml).  
- **YOLO_IMAGE_WIDTH**: *int* `<package>/dist/images`
> Image width of YOLO models input.  
> Normally 640.  
- **YOLO_IMAGE_HEIGHT**: *int* `<package>/dist/images`
> Image height of YOLO models input.  
> Normally 640.  
