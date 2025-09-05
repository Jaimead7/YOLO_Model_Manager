from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from torch import Tensor
from typing_extensions import Self
from ultralytics.engine.results import (OBB, Boxes, Keypoints, Masks, Probs,
                                        Results)
from ultralytics.utils.plotting import Annotator, Colors

from ..utils.config import RESULT_X_TOLERANCE, RESULT_Y_TOLERANCE


class Point(tuple):
    def __new__(cls, x: int, y: int) -> Self:
        x = int(x)
        y = int(y)
        return super().__new__(cls, (x, y))

    @property
    def x(self) -> int:
        return self[0]

    @property
    def y(self) -> int:
        return self[1]

    def distance(self, other: Point) -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def __add__(self, other) -> Point:
        if isinstance(other, (tuple, list)) and len(other) == 2:
            return Point(self[0] + other[0], self[1] + other[1])
        return Point(self[0] + other, self[1] + other)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.x}, {self.y})'


class Box(np.ndarray):
    def __new__(
        cls,
        input_array,
        conf: float,
        object_n: int
    ) -> Self:
        obj: Self = np.asarray(input_array).view(cls)
        obj.conf = conf
        obj.object_n = object_n
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.conf: float = getattr(obj, 'conf', 0.0)
        self.object_n: int = int(getattr(obj, 'object_n', 0))

    @property
    def sup_left_corner(self) -> Point:
        return Point(self[0], self[1])

    @property
    def sup_right_corner(self) -> Point:
        return Point(self[2], self[1])

    @property
    def inf_left_corner(self) -> Point:
        return Point(self[2], self[1])

    @property
    def inf_right_corner(self) -> Point:
        return Point(self[2], self[3])

    @property
    def min_x(self) -> int:
        return min(self[0], self[2])

    @property
    def max_x(self) -> int:
        return max(self[0], self[2])

    @property
    def min_y(self) -> int:
        return min(self[1], self[3])

    @property
    def max_y(self) -> int:
        return max(self[1], self[3])

    @property
    def center(self) -> Point:
        return self.get_center()

    @property
    def width(self) -> int:
        return self.sup_left_corner.x + self.inf_right_corner.x

    @property
    def height(self) -> int:
        return self.sup_left_corner.y + self.inf_right_corner.y

    def get_center(self) -> Point:
        x = int(self.width / 2)
        y = int(self.height / 2)
        return Point(x, y)

    def is_complete(
        self,
        img_w: int,
        img_h: int
    ) -> bool:
        return not any((
            self.min_x <= RESULT_X_TOLERANCE,
            self.max_x >= (img_w - RESULT_X_TOLERANCE),
            self.min_y <= RESULT_Y_TOLERANCE,
            self.max_y >= (img_h - RESULT_Y_TOLERANCE),
        ))

    def center_distance(self, other: Box) -> float:
        return self.center.distance(other.center)

    def add_square_to_img(self, img: np.ndarray, names: dict[int, str]) -> np.ndarray:
        #TODO: Use global params
        color: tuple = Colors()(self.object_n, bgr= True)
        text_color: tuple = Annotator(img).get_txt_color(color)
        text: str = f'{names[self.object_n]} {self.conf:.2f}'
        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.5
        text_thickness: int = 1
        border_thickness: int = 2
        (txt_w, txt_h), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness
        )
        text_sup_left_corner = Point(
            self.sup_left_corner.x - border_thickness,
            self.sup_left_corner.y + border_thickness
        )
        text_inf_right_corner = Point(
            self.sup_left_corner.x + txt_w + border_thickness,
            self.sup_left_corner.y - txt_h - border_thickness
        )
        cv2.rectangle(
            img= img,
            pt1= self.sup_left_corner,
            pt2= self.inf_right_corner,
            color= color,
            thickness= border_thickness
        )
        cv2.rectangle(
            img= img,
            pt1= text_sup_left_corner,
            pt2= text_inf_right_corner,
            color= color,
            thickness= -1
        )
        cv2.putText(
            img= img,
            text= text,
            org= self.sup_left_corner,
            fontFace= cv2.FONT_HERSHEY_SIMPLEX,
            fontScale= font_scale,
            color= text_color,
            thickness= text_thickness,
        )
        return img

    def add_center_to_img(self, img: np.ndarray) -> np.ndarray:
        color: tuple = Colors()(self.object_n, bgr= True)
        cv2.circle(img, self.center, 10, color, -1)
        return img


class MyBoxes(Boxes):
    def __init__(
        self,
        boxes: Tensor | np.ndarray | Boxes,
        orig_shape: tuple[int, int],
    ) -> None:
        if isinstance(boxes, Boxes):
            orig_shape = boxes.orig_shape
            boxes = boxes.data
        super().__init__(boxes, orig_shape)
        if isinstance(self.xyxy, Tensor):
            corners_array: np.ndarray = self.xyxy.cpu().numpy()
        else:
            corners_array: np.ndarray = self.xyxy
        if isinstance(self.conf, Tensor):
            conf_array: np.ndarray = self.conf.cpu().numpy()
        else:
            conf_array: np.ndarray = self.conf
        if isinstance(self.cls, Tensor):
            cls_array: np.ndarray = self.cls.cpu().numpy()
        else:
            cls_array: np.ndarray = self.cls
        self._boxes: list[Box] = [
            Box(*box)
            for box in zip(corners_array, conf_array, cls_array)
        ]
        self._completed_boxes: list[Box] = self.get_completed_boxes()
        self._valid_boxes: list[Box] = self.get_valid_boxes()

    @property
    def boxes(self) -> list[Box]:
        return self._boxes

    @property
    def completed_boxes(self) -> list[Box]:
        return self._completed_boxes

    @property
    def valid_boxes(self) -> list[Box]:
        return self._valid_boxes

    @property
    def centers(self) -> list[Point]:
        return [
            box.center
            for box in self.boxes
        ]

    def get_completed_boxes(self) -> list[Box]:
        return [
            box
            for box in self.boxes
            if box.is_complete(self.orig_shape[1], self.orig_shape[0])
        ]

    def get_valid_boxes(self) -> list[Box]:
        n: int = len(self.completed_boxes)
        valid_boxes: list[Box] = []
        for i in range(n):
            actual_box: Box = self.completed_boxes[i]
            is_alone = True
            is_best = True
            for j in range(n):
                if i != j:
                    if actual_box.center_distance(self.completed_boxes[j]) <= 20:
                        is_alone = False
                        if self.completed_boxes[j].conf >= actual_box.conf:
                            is_best = False
                            break
            if is_alone or is_best:
                valid_boxes.append(actual_box)
        return valid_boxes


class MyResults(Results):
    def __init__(self, result: Results) -> None:
        super().__init__(
            orig_img= result.orig_img,
            path= result.path,
            names= result.names
        )
        self.boxes = result.boxes
        self.masks: Optional[Masks] = result.masks
        self.keypoints: Optional[Keypoints] = result.keypoints
        self.probs: Optional[Probs | Tensor] = result.probs
        self.obb: Optional[OBB] = result.obb

    @property
    def img_w(self) -> int:
        return self.orig_img.shape[1]

    @property
    def img_h(self) -> int:
        return self.orig_img.shape[0]

    @property
    def boxes(self) -> Optional[MyBoxes]:
        return self._boxes

    @boxes.setter
    def boxes(self, boxes: Optional[Boxes]) -> None:
        if boxes is None:
            self._boxes: Optional[MyBoxes] = None
        else:
            self._boxes: Optional[MyBoxes] = MyBoxes(boxes, boxes.orig_shape)

    @property
    def completed_boxes(self) -> list[Box]:
        if self.boxes is None:
            return []
        return self.boxes.completed_boxes

    @property
    def valid_boxes(self) -> list[Box]:
        if self.boxes is None:
            return []
        return self.boxes.valid_boxes

    def plot_tracker(self) -> np.ndarray:
        img: np.ndarray = self.orig_img
        if self.boxes is not None:
            for box in reversed(self.boxes.boxes):
                img = box.add_square_to_img(img, self.names)
                img = box.add_center_to_img(img)
        return self.plot()


class ResultTracker:
    MAX_RESULTS = 5
    def __init__(self) -> None:
        self.results_hist: list[MyResults] = []

    def add_new_result(self, new_result: Results) -> None:
        self.results_hist.append(MyResults(new_result))
        if len(self.results_hist) > self.MAX_RESULTS:
            self.results_hist.pop(0)

    def plot(self) -> np.ndarray:
        return self.results_hist[-1].plot_tracker()
