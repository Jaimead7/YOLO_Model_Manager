import cv2
import numpy as np
from ultralytics.engine.results import Results

class MyResult(Results):
    def __init__(self, result: Results) -> None:
        self.__dict__.update(result.__dict__)
        
    def plot_with_centers(self) -> np.ndarray:
        result_img: np.ndarray = self.plot()
        if self.boxes is not None:
            for box in self.boxes.xyxy.cpu().numpy():  # type: ignore
                x = int((box[0] + box[2]) / 2)
                y = int((box[1] + box[3]) / 2)
                cv2.circle(result_img, (x, y), 10, (0, 255, 0), -1)
        return result_img
