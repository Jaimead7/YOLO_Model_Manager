# [Image Processing](../../../yoloModelManager/src/image/image_processing.py)  
Provides the class [ImageProcessing](../../../yoloModelManager/src/image/image_processing.py#L18) for image management as a np.ndarray.  
The class groups the following functions:
- Filters `fun(img: np.ndarray, ...) -> np.ndarray`:  
    - [bgr2gray](#bgr2gray)  
    - [gray2bgr](#gray2bgr)  
    - [resize](#resize)  
    - [cut](#cut)  
    - [border](#border)  
    - [padding](#padding)  
- Images lists `fun(img: list[np.ndarray], ...) -> list[np.ndarray]`:  
    - [unify_images](#unify_images)  
    - [set_images_as_rgb](#set_images_as_rgb)  
    - [unify_shapes](#unify_shapes)  
- [FILTERS](#filters)  
- [get_filter_name](#get_filter_name)  
- [get_images_grid](#get_images_grid)  

</br>

## [bgr2gray](../../../yoloModelManager/src/image/image_processing.py#L21)  
**`bgr2gray(img: np.ndarray)`** -> *np.ndarray*  
Turn a color image into a gray image.  

</br>

## [gray2bgr](../../../yoloModelManager/src/image/image_processing.py#L27)  
**`gray2bgr(img: np.ndarray)`** -> *np.ndarray*  
Turn a gray image into a color image.  

</br>

## [resize](../../../yoloModelManager/src/image/image_processing.py#L33)  
**`resize(img: np.ndarray, width: int = YOLO_IMAGE_WIDTH, height: int = YOLO_IMAGE_HEIGHT)`** -> *np.ndarray*  
Transform the image into the new dimensions.  
The image will be streched.  

</br>

## [cut](../../../yoloModelManager/src/image/image_processing.py#L45)  
**`cut(img: np.ndarray, width: int = YOLO_IMAGE_WIDTH, height: int = YOLO_IMAGE_HEIGHT)`** -> *np.ndarray*  
Transform the image into the new dimensions.  
The image will be cuted.  
Only if the new shape is smaller than the current.  

</br>

## [border](../../../yoloModelManager/src/image/image_processing.py#L54)  
**`border(img: np.ndarray, width: int = 1, color: tuple[int, int, int, int] = (255, 255, 255, 255))`** -> *np.ndarray*  
Add an internal border to the image.  

</br>

## [padding](../../../yoloModelManager/src/image/image_processing.py#L70)  
**`padding(img: np.ndarray, target_height: int, target_width: int, color: tuple[int, int, int, int] = (255, 255, 255, 255))`** -> *np.ndarray*  
Transform the image into the new dimensions.  
Adds a plain color to complete the shape.  
Only if the new shape is bigger than the current.  

</br>

## [unify_images](../../../yoloModelManager/src/image/image_processing.py#L116)  
**`unify_images(images: list[np.ndarray])`** -> *np.ndarray*  
Transform all the images into rgb and same shape with [padding](#padding).  
The final shape will be the maximum of width and height of the images of the list.  

</br>

## [set_images_as_rgb](../../../yoloModelManager/src/image/image_processing.py#L125)  
**`set_images_as_rgb(images: list[np.ndarray])`** -> *np.ndarray*  
Transform all the images into rgb.  

</br>

## [unify_shapes](../../../yoloModelManager/src/image/image_processing.py#L138)  
**`unify_shapes(images: list[np.ndarray])`** -> *np.ndarray*  
Transform all the images into same shape with [padding](#padding).  
The final shape will be the maximum of width and height of the images of the list.  

</br>

## [FILTERS](../../../yoloModelManager/src/image/image_processing.py#L93)  
`dict[str, Callable]` with a filter name and the filter func.  

</br>

## [get_filter_name](../../../yoloModelManager/src/image/image_processing.py#L103)  
**`get_filter_name(filter: Callable)`** -> *str*  
Returns the name of the filter func.  
`filter` should be obtained by `ImageProcessing.FILTERS[filter]` not an import. 

</br>

## [get_images_grid](../../../yoloModelManager/src/image/image_processing.py#L151)  
**`get_images_grid(images: list[np.ndarray])`** -> *np.ndarray*  
Returns an image with all the images in the list in a mosaic.  
