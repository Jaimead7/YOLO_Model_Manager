from math import ceil, sqrt
from typing import Callable

import cv2
import numpy as np

from ..utils.config import YOLO_IMAGE_HEIGHT, YOLO_IMAGE_WIDTH, my_logger


class ImageProcessing:
    #---------- FILTERS ----------#
    @staticmethod
    def bgr2gray(
        img: np.ndarray
    ) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def gray2bgr(
        img: np.ndarray
    ) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def resize(
        img: np.ndarray,
        width: int = YOLO_IMAGE_WIDTH,
        height: int = YOLO_IMAGE_HEIGHT
    ) -> np.ndarray:
        return cv2.resize(
            img,
            (width, height),
            interpolation= cv2.INTER_LINEAR
        )

    @staticmethod
    def cut(
        img: np.ndarray,
        width: int = YOLO_IMAGE_WIDTH,
        height: int = YOLO_IMAGE_HEIGHT
    ) -> np.ndarray:
        ... #TODO: Do cut filter
        return img

    @staticmethod
    def border(
        img: np.ndarray,
        width: int = 1,
        color: tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> np.ndarray:
        return cv2.copyMakeBorder(
            img,
            width,
            width,
            width,
            width,
            cv2.BORDER_CONSTANT,
            value= color
        )

    @staticmethod
    def padding(
        img: np.ndarray,
        target_height: int,
        target_width: int,
        color: tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> np.ndarray:
        height: int
        width: int
        height, width = img.shape[:2]
        delta_h: int = target_height - height
        delta_h = delta_h if delta_h >= 0 else 0
        delta_w: int = target_width - width
        delta_w = delta_w if delta_w >= 0 else 0
        return cv2.copyMakeBorder(
            img,
            delta_h // 2,
            delta_h - (delta_h // 2),
            delta_w // 2,
            delta_w - (delta_w // 2),
            cv2.BORDER_CONSTANT,
            value= color
        )

    FILTERS: dict[str, Callable] = {
        'GREY': bgr2gray,
        'COLOR': gray2bgr,
        'RESIZE': resize,
        'CUT': cut,
        'BORDER': border,
        'PADDING': padding
    }

    @classmethod
    def get_filter_name(
        cls,
        filter: Callable
    ) -> str:
        filters: dict[Callable, str] = {
            func: name
            for name, func in cls.FILTERS.items()
        }
        return filters[filter]
    #-----------------------------#

    #---------- IMAGES LISTS ----------#
    @classmethod
    def unify_images(
        cls,
        images: list[np.ndarray]
    ) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = cls.set_images_as_rgb(images)
        unified_images = cls.unify_shapes(unified_images)
        return unified_images

    @classmethod
    def set_images_as_rgb(
        cls,
        images: list[np.ndarray]
    ) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = []
        for image in images:
            if len(image.shape) == 2:
                unified_images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            else:
                unified_images.append(image)
        return unified_images

    @classmethod
    def unify_shapes(
        cls,
        images: list[np.ndarray]
    ) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = []
        maxW: int = max(image.shape[1] for image in images)
        maxH: int = max(image.shape[0] for image in images)
        for image in images:
            unified_images.append(cls.padding(image, maxH, maxW))
        return unified_images
    #----------------------------------#

    @classmethod
    def get_images_grid(
        cls,
        images: list[np.ndarray]
    ) -> np.ndarray:
        images = cls.unify_images(images)
        images = [cls.border(image) for image in images]
        cols: int = ceil(sqrt(len(images)))
        rows: list[np.ndarray] = []
        for row in range(0, len(images), cols):
            row_images: list[np.ndarray] = images[row:row+cols]
            if len(row_images) < cols:
                empty_image: np.ndarray = np.zeros_like(images[0])
                row_images.extend([empty_image] * (cols - len(row_images)))
            rows.append(cv2.hconcat(row_images))
        return cv2.vconcat(rows)
