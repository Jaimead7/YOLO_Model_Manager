
# Split Dataset Command  
Create a directory structure of train, validation and test datasets from a [label-studio](https://labelstud.io/) dataset. The dataset needs to be exported with a [YOLO structure](https://labelstud.io/guide/export#YOLO) (with or without images).  
The directory structure is ready to use with [Ultralitycs YOLO11](https://docs.ultralytics.com/es/models/yolo11/) models.  

## Usage:
```bash
split-dataset [OPTIONS]
```  
| OPTION | VALUE | DESCRIPTION |
|-|-|-|
-d, --data-source | TEXT | Path to the exported dataset from label studio. If path is relative, relative to environment variable `IMAGES_SAVE_PATH` or `app/images`. `Train`/`Validation`/`Test` directories will be created on parent. It can be a directory or a zip file. `[required]`.  
-i, --images | PATH | Path to the directory with the images. Only needed if the dataset is exported without images.  
-v, --validation | FLOAT RANGE | % of the images for validation. Default to 0.2. `[0<x<1]`  
-t, --test | FLOAT RANGE | % of the images for test. Default to 0.1. `[0<x<1]`  
--help | | Show this message and exit.  

## Dataset direcotry structure:  
```
label_studio_dataset -> /, .zip, .tar, .gz, .bz2, .xz
|--- images
|    |--- image1.png
|    |--- image2.png
|    `--- ...
|--- labels
|    |--- image1.txt
|    |--- image2.txt
|    `--- ...
|--- classes.txt
`--- notes.json -> Optional
```
```
splited_dataset
|--- train
|    |--- images
|    |    |--- image2.png
|    |    |--- image7.png
|    |    `--- ...
|    `--- labels
|         |--- image2.txt
|         |--- image7.txt
|         `--- ...
|--- validation
|    |--- images
|    |    |--- image9.png
|    |    |--- image1.png
|    |    `--- ...
|    `--- labels
|         |--- image9.txt
|         |--- image1.txt
|         `--- ...
|--- test
|    |--- images
|    |    |--- image3.png
|    |    |--- image6.png
|    |    `--- ...
|    `--- labels
|         |--- image3.txt
|         |--- image6.txt
|         `--- ...
|--- metadata.yaml
`--- data.yaml
```

### Files examples:  
- [metadata.yaml](../examples/model.metadata.yaml)  
- [data.yaml](../examples/dataset.data.yaml)  
