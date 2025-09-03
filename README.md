# YOLO_Model_Manager
Managing custom YOLO models training and execution.

## Contact:  
> **Jaime Ávarez Díaz**  
> *e-mail:* alvarez.diaz.jaime1@gmail.com  


## Description:  
This is a CLI application to train the AI models.  

## Instalation:  
Python >=3.10 <=3.12 needed.  
### Windows *(with Powershell)*:  
```powershell
cd <repoDir>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install .
#py -m pip install -e . #for development
```

### Linux *(with bash)*:  
```bash
cd <repoDir>
python -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip
pip install .
#pip install -e . #for development
```

### Crate environment variables:  
Create [`./yoloModelManager/dist/.env`](./yoloModelManager/dist/) env file.  
Check the structure of the file in [**`example.env`**](./docs/examples/example.env).  
Environment variables can be loaded before importing anything from the package.  
#### Example:
```python
from dotenv import load_dotenv
load_dotenv('example/path/.env')
from yoloModelManager.utils import *
```

## Usage:
Activate the virtual environment before using any scripts.  

The CLI provides the following commands:
- [**Image adquisition:**](./docs/cli/image-adquisition) `image-adquisition [OPTIONS]`
- [**Test model:**](./docs/cli/test-model) `test-model [OPTIONS]`
- [**Train model**](./docs/cli/train-model) `train-model [OPTIONS]`
- [**Split dataset**](./docs/cli/split-dataset) `split-dataset [OPTIONS]`
- [**Move Cursor**](./docs/cli/move-cursor) `move-cursor [OPTIONS]`

## License:
This package has a [AGPL-3.0](LICENSE) license due to the usage of [Ultralytics](https://github.com/ultralytics/ultralytics/) package.
