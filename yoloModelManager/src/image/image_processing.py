from math import ceil, sqrt
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
from pyUtils import MyLogger, Styles

from ..utils import (IMAGE_LOGGING_LVL, IMAGES_PATH, YOLO_IMAGE_HEIGHT,
                     YOLO_IMAGE_WIDTH)

my_logger = MyLogger(f'{__name__}', IMAGE_LOGGING_LVL)


class ImageProcessing:
    @staticmethod
    def bgr2gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def gray2bgr(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def cut(
        frame: np.ndarray,
        width: int = YOLO_IMAGE_WIDTH,
        height: int = YOLO_IMAGE_HEIGHT
    ) -> np.ndarray:
        ... #TODO
        return frame

    @staticmethod
    def resize(
        frame: np.ndarray,
        width: int = YOLO_IMAGE_WIDTH,
        height: int = YOLO_IMAGE_HEIGHT
    ) -> np.ndarray:
        return cv2.resize(
            frame,
            (width, height),
            interpolation= cv2.INTER_LINEAR
        )

    @staticmethod
    def add_border(
        frame: np.ndarray,
        width: int = 1,
        color: tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> np.ndarray:
        return cv2.copyMakeBorder(
            frame,
            width,
            width,
            width,
            width,
            cv2.BORDER_CONSTANT,
            value= color
        )

    @classmethod
    def unify_images(cls, images: list[np.ndarray]) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = cls.unify_colors(images)
        unified_images = cls.unify_shapes(unified_images)
        return unified_images

    @staticmethod
    def unify_colors(images: list[np.ndarray]) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = []
        for image in images:
            if len(image.shape) == 2:
                unified_images.append(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            else:
                unified_images.append(image)
        return unified_images

    @classmethod
    def unify_shapes(cls, images: list[np.ndarray]) -> list[np.ndarray]:
        unified_images: list[np.ndarray] = []
        maxW: int = max(image.shape[1] for image in images)
        maxH: int = max(image.shape[0] for image in images)
        for image in images:
            unified_images.append(cls.imagePadding(image, maxH, maxW))
        return unified_images

    @staticmethod
    def imagePadding(
        image: np.ndarray,
        target_height: int,
        target_width: int,
        color: tuple[int, int, int, int] = (255, 255, 255, 255)
    ) -> np.ndarray:
        height: int
        width: int
        height, width = image.shape[:2]
        delta_h: int = target_height - height
        delta_w: int = target_width - width
        return cv2.copyMakeBorder(
            image,
            delta_h // 2,
            delta_h - (delta_h // 2),
            delta_w // 2,
            delta_w - (delta_w // 2),
            cv2.BORDER_CONSTANT,
            value= color
        )

    @classmethod
    def get_images_grid(cls, images: list[np.ndarray]) -> np.ndarray:
        images = cls.unify_images(images)
        images = [cls.add_border(image) for image in images]
        cols: int = ceil(sqrt(len(images)))
        rows: list[np.ndarray] = []
        for row in range(0, len(images), cols):
            row_images: list[np.ndarray] = images[row:row+cols]
            if len(row_images) < cols:
                empty_image: np.ndarray = np.zeros_like(images[0])
                row_images.extend([empty_image] * (cols - len(row_images)))
            rows.append(cv2.hconcat(row_images))
        return cv2.vconcat(rows)

    @classmethod
    def save_image(cls, image: np.ndarray, dir_path: Optional[Path] = None) -> Path:
        if dir_path is None:
            dir_path = IMAGES_PATH
        image_name: str = f'{uuid4()}.png'
        image_path: Path = dir_path / image_name
        if cv2.imwrite(str(image_path), image):
            my_logger.debug(f'New image saved to "{image_path}"', Styles.SUCCEED)
        else:
            msg: str = f'Failed to save image to "{image_path.parent}". Check if directory exists.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        return image_path

    @classmethod
    def get_filter_name(cls, filter: Callable) -> str:
        filters: dict[Callable, str] = {
            func: name
            for name, func in cls.FILTERS.items()
        }
        return filters[filter]

    FILTERS: dict[str, Callable] = {
        'GREY': bgr2gray,
        'COLOR': gray2bgr,
        'RESIZE': resize,
        'CUT': cut
    }
