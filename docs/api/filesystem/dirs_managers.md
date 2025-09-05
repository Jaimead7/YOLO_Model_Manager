# Directories Managers
Documentation for [dirs_managers.py](../../../yoloModelManager/src/filesystem/dirs_managers.py)
- [DatasetDirManager](#datasetdirmanager)
-  [TrainingDatasetDirManager](#trainingdatasetdirmanager)

<br>

## DatasetDirManager
Class for managing a directory containing labeled images.  
This follows the directory structure exported by label-studio.  
### Directory structure
```
Image_Dir
|--- images
|    |--- metadata.yaml
|    |--- image1.png
|    |--- image2.png
|    `--- ...
|--- labels
|    |--- image1.txt
|    |--- image2.txt
|     `--- ...
|--- metadata.yaml
`--- ...
```
### Attributes
- **path**: *Path*  
> Absolute path of the dir.  
> The `setter` admits a `str | Path` object.  
- **images_path**: *Path* `Read-only`  
> Absolute path of the images subdir.  
- **labels_path**: *Path* `Read-only`  
> Absolute path of the labels subdir.  
- **metadata_path**: *Path* `Read-only`  
> Absolute path of the metadata of the images.  
- **create**: *bool*  
> Whether to create the directory when `path` is setted.  
### Methods
- **__init__(path: Path, create: bool = True)** -> *None*
> Create the [**DatasetDirManager**](#datasetdirmanager) object with the `path` dir.  
If `create`, create the directory if it doesn't exists.  
- **get_images_list()** -> *list[Path]*  
> List of the absolute paths of the images files in th images dir.  
- **get_labels_list()** -> *list[Path]*
> List of the absolute paths of the labels files in th labels dir.  
- **add_data(images: list[Path], labels: list[Path])** -> *None*
> Copy the images files in `images` in it's images directory and the labels files in `labels` in it's labels dir.  
- **add_images(imagesPath: Path)** -> *None*
> Copy the images files in the directory `imagesPath` in it's images dir.  
- **get_n_images()** -> *int*  
> Return the number of images in the folder.  

<br>

## TrainingDatasetDirManager
Class for managing a directory containing three [**DatasetDirManager**](#datasetdirmanager).  
Each of this directories  contains the corresponding train, validation and test images and labels.  
The directory has a `.yaml` file with key data for training the model.  
A complete dataset for training a model.  
### Directory structure
```
TrainingDir
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
### `.yaml` file structure
- Check [metadata.yaml](../../examples/dataset.metadata.yaml).  
- Check [data.yaml](../../examples/dataset.data.yaml).  

### Attributes
- **source_dataset_dir**: *DatasetDirManager*  
> [**DatasetDirManager**](#datasetdirmanager) object for the source data.  
> The `setter` admits a `Path` object and creates the [**DatasetDirManager**](#datasetdirmanager) object (It doesn\`t creates the directory if it doesn\`t exists).  
> If the directory is a zip directory it will unzip it.  
> Supported extensions: `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`.  
> If the `Path` is relative it is relative to `DATASETS_PATH`.  
- **path**: *Path*  
> Absolute path of the dir.  
> The default value is: `<source_dataset_path>.parent / <datasetName>`.  
- **dataset_name**: *str* `Read-only`  
> Name of the dir.  
> The default name is: `<source_dataset_name>_split`.  
> Creates the directory if it doesn`t exist.  
- **train_dir**: *[DatasetDirManager](#datasetdirmanager)*  
> [**DatasetDirManager**](#datasetdirmanager) object for the train subdir.  
- **validation_dir**: *[DatasetDirManager](#datasetdirmanager)*  
> [**DatasetDirManager**](#datasetdirmanager) object for the validation subdir.  
- **test_dir**: *[DatasetDirManager](#datasetdirmanager)*  
> [**DatasetDirManager**](#datasetdirmanager) object for the tests subdir.  
- **data_yaml_file_path**: *Path*  
> Absolute path of the `data.yaml` file.  
- **data**: [ModelTrainingDataDict](../utils/data_types.md/#modeltrainingdatadict)  
> Content of the `data.yaml` file.  
- **metadata_yaml_file_path**: *Path*  
> Absolute path of the `metadata.yaml` file.  
- **metadata**: [DatasetMetadataDict](../utils/data_types.md/#datasetmetadatadict)  
> Content of the `data.yaml` file.  

### Methods
- **set_paths()** -> *None*  
> Set the paths of the directory and subdirectories.  
> Doesn't create the structure.  
- **create_paths()** -> *None*  
> Set the paths of the directory and subdirectories.  
> Creates the directory structure if it doesn`t exist.  
- **split(validation: float = 0.2, test: float = 0.1)** -> *None*
> Split the source dataset into the train, validation and test datasets with a percentage of `validation` and `test` for validation and test datasets.  
- **create_yaml_data_file()** -> *None*
> creates the `data.yaml` file.  
- **get_n_train()** -> *int*  
> Return the number of images in the train folder.  
- **get_n_val()** -> *int*  
> Return the number of images in the validation folder.  
- **get_n_test()** -> *int*  
> Return the number of images in the test folder.  