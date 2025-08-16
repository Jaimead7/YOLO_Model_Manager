# Directories utilities
Documentation for [dirs.py](../../../yoloModelManager/src/filesystem/dirs.py)
- [unzip_dir](#unzip_dir)
- [check_dir_path](#check_dir_path)

<br>

## unzip_dir
Unzip a file in the current parent folder.  

### Arguments
- **dir**: *Path*
> Path of the file to unzip.

### Raises
- **FileExistsError**
> If file doesn't exists.

- **Value error**
> If file has no zip extension.  
> Abailable extensions: `.zip`, `.tar`, `.gz`, `.bz2`, `.xz`

### Returns
`Path` of the unziped directory.

<br>

## check_dir_path
Check if a path is a directory.  
If `create` the directory will be created if it doesent exists.

### Arguments
- **path_in**: *Any*
> Path like of the directory to check.

- **create**: *bool* -> `True`
> Whether to create the folder if it doesn't exists or not.

### Raises
- **TypeError**
> If `path_in` is not a path like.

- **NotADirectoryError**
> If the path is not a directory and `create` is not true.

### Returns
`Path` of the directory.