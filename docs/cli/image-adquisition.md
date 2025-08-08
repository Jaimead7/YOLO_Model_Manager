# Image Adquisition Command  
Camera streaming and save images to the disc.  

## Usage:
```bash
image-adquisition [OPTIONS]
```  
| OPTION | VALUE | DESCRIPTION |
|-|-|-|
-c, --camera | INTEGER RANGE | Id of the camera for opencv. Defaults to None for select. `[x>=0]`  
-f, --show-filter | [grey \| color \| resize \| cut] | Processing filter for showing images. Valid options: `['GREY', 'COLOR', 'RESIZE', 'CUT']`  
-s, --save-filter | [grey \| color \| resize \| cut] | Processing filter for saving images. Valid options: `['GREY', 'COLOR', 'RESIZE', 'CUT']`  
-p, --save-path | PATH | Path to save the file. If `None` try to import from environment variable `IMAGES_SAVE_PATH`. Else set to app/images.  
--help | | Show this message and exit.

## Kewboard shortcuts:  
- ```ESC```: Exit the program.  
- ```SPACE```: Save frame.  
