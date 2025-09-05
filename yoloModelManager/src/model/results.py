from math import sqrt

import cv2
import numpy as np
from ultralytics.engine.results import Results, Boxes


class ResultTracker:
    MAX_RESULTS = 5
    def __init__(self) -> None:
        self.results_hist: list[MyResult] = []

    def add_new_result(self, new_result: Results) -> None:
        self.results_hist.append(MyResult(new_result))
        if len(self.results_hist) > self.MAX_RESULTS:
            self.results_hist.pop(0)
        ...

    def plot(self) -> np.ndarray:
        img: np.ndarray = self.results_hist[-1].plot_centers()
        try:
            p1: tuple[int, int] = self.results_hist[-1].centers[0]
            p2: tuple[int, int] = self.results_hist[0].centers[0]
            cv2.line(img, p1, p2, (255, 0, 0), 5)
        except IndexError:
            ...
        return img


class MyResult(Results):
    COLORS: tuple[tuple[int, int, int], ...] = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 165, 0),
        (128, 0, 128),
        (0, 128, 0),
        (255, 192, 203)
    )
    
    def __init__(self, result: Results) -> None:
        self.__dict__.update(result.__dict__)

    @property
    def img_w(self) -> int:
        return self.orig_img.shape[1]

    @property
    def img_h(self) -> int:
        return self.orig_img.shape[0]

    @property
    def valid_boxes(self) -> list[Boxes]:
        return [
            box
            for box in self.completed_boxes
            if self.box_is_valid(box)
        ]

    @property
    def completed_boxes(self) -> list[Boxes]:
        if self.boxes is None:
            return []
        return [
            box
            for box in self.boxes
            if self.box_is_complete(box)
        ]

    @property
    def centers(self) -> list[tuple[int, int]]:
        return [
            self.get_center(box)
            for box in self.completed_boxes
        ]

    @staticmethod
    def get_center(box: Boxes) -> tuple[int, int]:
        box_array: np.ndarray = box.xyxy.cpu().numpy()[0]  # type: ignore
        x = int((box_array[0] + box_array[2]) / 2)
        y = int((box_array[1] + box_array[3]) / 2)
        return (x, y)

    @staticmethod
    def distance_of_centers(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0])

    def plot_centers(self) -> np.ndarray:
        result_img: np.ndarray = self.plot()
        for i, center in enumerate(self.centers):
            cv2.circle(result_img, center, 10, self.COLORS[i], -1)
        return result_img

    def box_is_valid(self, box: Boxes) -> bool:
        if box not in self.completed_boxes:
            return False
        center: tuple[int, int] = self.get_center(box)
        for other_box in self.completed_boxes:
            if self.distance_of_centers(center, self.get_center(other_box)) < 20:
                if box.conf < other_box.conf:
                    return False
        return True

    def box_is_complete(self, box: Boxes) -> bool:
        box_array: np.ndarray = box.xyxy.cpu().numpy()[0]  # type: ignore
        TOLERANCE: int = 2
        return not any((
            box_array[0] <= TOLERANCE,
            box_array[2] >= (self.img_w - TOLERANCE),
        ))
