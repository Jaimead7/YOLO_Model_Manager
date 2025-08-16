# Files utilities
Documentation for [files.py](../../../yoloModelManager/src/filesystem/files.py)
- [ALLOWED_IMAGES_EXTENSIONS](#allowed_images_extensions)
- [copy_files](#copy_files)

<br>

## ALLOWED_IMAGES_EXTENSIONS
List of the extensions allowed for images files.

<br>

## copy_files
Copy a list of files into a directory.  

### Arguments
- **files_list**: *list[Path]*
> List of the Paths of the files to copy.
- **destiny_dir**: *Path*
> Derectory to copy the files into.
- **new_names**: *Optional[list[str]]* `None`
> List of the new names of the files.  
> The order of the new names should be the same of the `file_list`

### Raises
- **NotADirectoryError**
> If file destiny directory doesn't exists.

### Returns
`None`
