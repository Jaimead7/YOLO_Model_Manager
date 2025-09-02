import platform
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from pprint import pprint
from sys import exit
from typing import Any, Callable, Generator, Optional, TypedDict

import cv2
import numpy as np
from pyUtils import MyLogger, time_me

from ..image.image_processing import ImageProcessing
from ..model.data import create_model_medatada_yaml
from ..model.model_manager import ModelManager
from ..utils.config import CAMERA_LOGGING_LVL, IMAGES_PATH

my_logger = MyLogger(
    logger_name= f'{__name__}',
    logging_level= CAMERA_LOGGING_LVL,
    file_path= 'yoloModelManager.log',
    save_logs= False
)


class CameraInfo(TypedDict):
    index: int
    name: str
    width: int
    height: int


class CameraManager(ABC):
    def __init__(
        self,
        camera_id: Optional[int] = None
    ) -> None:
        self.camera_info: CameraInfo = self.select_camera(camera_id)
        with self.get_video_capture() as cap:
            self.width, self.height = self.get_camera_resolution(cap)
        self.show_filters = None
        self.save_filters = None
        self.save_dir_path = None
        self.keys_callbacks: dict[int, tuple[Callable, dict]] = {} #TODO: add parameters to callbacks
        my_logger.info(f'Camera set to: {self.camera_info}.')
        super().__init__()

    @property
    def camera(self) -> int:
        return self.camera_info['index']

    @property
    def name(self) -> str:
        return self.camera_info['name']

    @property
    def width(self) -> int:
        return self.camera_info['width']

    @width.setter
    def width(self, w: int) -> None:
        self.camera_info['width'] = w

    @property
    def height(self) -> int:
        return self.camera_info['height']

    @height.setter
    def height(self, h: int) -> None:
        self.camera_info['height'] = h

    @property
    def show_filters(self) -> list[Callable]:
        if self._show_filters is None:
            return []
        return self._show_filters

    @show_filters.setter
    def show_filters(self, filters: Optional[list[Callable]]) -> None:
        #TODO: check filters
        self._show_filters: Optional[list[Callable]] = filters

    @property
    def save_filters(self) -> list[Callable]:
        if self._save_filters is None:
            return []
        return self._save_filters

    @save_filters.setter
    def save_filters(self, filters: Optional[list[Callable]]) -> None:
        #TODO: check filters
        self._save_filters: Optional[list[Callable]] = filters

    @property
    def save_dir_path(self) -> Path:
        if self._save_dir_path is None:
            return IMAGES_PATH
        return self._save_dir_path

    @save_dir_path.setter
    def save_dir_path(self, path: Optional[str | Path]) -> None:
        if path is None:
            self._save_dir_path = None
        else:
            self._save_dir_path = Path(path)
            if not self._save_dir_path.is_dir():
                self._save_dir_path.mkdir(
                    parents= True
                )
                my_logger.info(f'"{self._save_dir_path}" created.')
        my_logger.debug(f'Save path of camera "{self.camera_info["index"]}": "{self.save_dir_path}".')

    @staticmethod
    @abstractmethod
    @time_me
    def get_cameras_info() -> list[dict]:
        raise NotImplementedError()

    @staticmethod
    @time_me(debug= False)
    def camera_exists(camera_id: int) -> bool:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            cap.release()
            return True
        return False

    @classmethod
    @time_me
    def detect_working_cameras(cls, max_to_check= 5) -> list[int]:
        working_cameras: list = []
        for i in range(max_to_check):
            if cls.camera_exists(i):
                working_cameras.append(i)
        return working_cameras

    @classmethod
    @time_me
    def get_cameras(cls) -> dict[int, CameraInfo]:
        cameras_info: list[dict] = cls.get_cameras_info()
        working_indices: list[int] = cls.detect_working_cameras()
        if not working_indices:
            msg: str = 'Cameras not found.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        cameras: dict[int, CameraInfo] = {}
        #FIXME: the order of getCamerasInfo is not the same that detectWorkingCameras so the names may not be ok.
        for cv2_index in working_indices:
            cameras[cv2_index] = CameraInfo(
                index= cv2_index,
                name= cameras_info[cv2_index]['Name'],
                width= 0, #TODO: change to cameras_info[cv2_index]['resolution']
                height= 0
            )
        return cameras

    @classmethod
    def select_camera(
        cls,
        camera_id: Optional[int] = None
    ) -> CameraInfo:
        cameras: dict[int, CameraInfo] = cls.get_cameras()
        try:
            if camera_id is not None:
                for camera in cameras.values():
                    if camera['index'] == camera_id:
                        return camera
            print('Available cameras:')
            pprint(cameras)
            user_input: str = input('Select a camera index: ')
            return cameras[int(user_input)]
        except (KeyError, ValueError):
            msg: str = 'Camera index not valid.'
            my_logger.critical(msg)
            exit(1)

    @contextmanager
    def get_video_capture(self) -> Generator[cv2.VideoCapture, Any, None]:
        cap: cv2.VideoCapture = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            msg: str = 'Can\'t connect to the camera.'
            my_logger.error(f'ConnectionRefusedError: {msg}')
            raise ConnectionRefusedError(msg)
        try:
            yield cap
        finally:
            cap.release()

    @time_me
    def get_camera_resolution(self, cap: cv2.VideoCapture) -> tuple[int, int]:
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        my_logger.info(f'Camera resolution: {self.width}x{self.height} px.')
        return (self.width, self.height)

    @time_me
    def set_camera_resolution(
        self,
        w: int,
        h: int,
        cap: cv2.VideoCapture
    ) -> tuple[int, int]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return self.get_camera_resolution(cap)

    @time_me
    def reset_window_to_camera_resolution(self) -> None:
        cv2.resizeWindow(self.name, self.width, self.height)
        my_logger.debug(f'Window resize to: {self.width}x{self.height} px.')

    @staticmethod
    def get_frame(
        cap: cv2.VideoCapture
    ) -> np.ndarray:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            msg: str = 'Can\'t read frame.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        return frame

    def exit(self, *args, **kwargs) -> int:
        my_logger.info('Stopping model stream...')
        create_model_medatada_yaml(
            self.save_dir_path,
            self.width,
            self.height,
            [ImageProcessing.get_filter_name(filter)
             for filter in self.save_filters]
        )
        return -1

    def save_last_frame(self, *args, **kwargs) -> int:
        try:
            subfolder = Path(kwargs['subfolder'])
        except:
            subfolder = Path("")
        if len(self.save_filters) == 0:
            ImageProcessing.save_image(self.last_frame, self.save_dir_path / subfolder)
        else:
            frame: np.ndarray = self.last_frame
            for filter in self.save_filters:
                frame: np.ndarray = filter(frame)
            ImageProcessing.save_image(frame, self.save_dir_path / subfolder)
        return 0

    def video_stream(
        self,
        width: int = 640,
        height: int = 480
    ) -> None:
        self.keys_callbacks[27] = (self.exit, {})
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        with self.get_video_capture() as cap:
            self.set_camera_resolution(width, height, cap)
            self.reset_window_to_camera_resolution()
            my_logger.debug('Starting video stream.')
            while True:
                self.last_frame: np.ndarray = self.get_frame(cap)
                frames: list[np.ndarray] = [self.last_frame]
                for filter in self.show_filters:
                    frames.append(filter(self.last_frame))
                images_grid: np.ndarray = ImageProcessing.get_images_grid(frames)
                cv2.imshow(self.name, images_grid)
                key: int = cv2.waitKey(1)
                try:
                    if self.keys_callbacks[key][0](**self.keys_callbacks[key][1]) < 0:
                        break
                except KeyError:
                    if key != -1:
                        my_logger.debug(f'Key pressed: "{key}".')
        cv2.destroyAllWindows()

    def model_stream(
        self,
        model: ModelManager
    ) -> None:
        self.keys_callbacks[27] = (self.exit, {})
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        self.save_filters = model.filters
        with self.get_video_capture() as cap:
            self.set_camera_resolution(model.camera_width, model.camera_height, cap)
            self.reset_window_to_camera_resolution()
            my_logger.debug('Starting video stream.')
            while True:
                self.last_frame: np.ndarray = self.get_frame(cap)
                model.process_frame(self.last_frame)
                frame_compose: np.ndarray = model.get_last_result_image()
                cv2.imshow(self.name, frame_compose)
                key: int = cv2.waitKey(1)
                try:
                    if self.keys_callbacks[key][0](**self.keys_callbacks[key][1]) < 0:
                        break
                except KeyError:
                    if key != -1:
                        my_logger.debug(f'Key pressed: {key}')
        cv2.destroyAllWindows()


class WindowsCameraManager(CameraManager):
    @staticmethod
    @time_me
    def get_cameras_info() -> list[dict]:
        import wmi  # type: ignore
        cameras: list[dict] = []
        for device in wmi.WMI().query('SELECT * FROM Win32_PnPEntity'):
            if device.PNPClass == 'Camera':
                device_props: dict = {}
                for prop in device._properties:
                    value: Any = getattr(device, prop)
                    device_props[prop] = value
                cameras.append(device_props)
        return cameras


class LinuxCamerasManager(CameraManager):
    @staticmethod
    @time_me
    def get_cameras_info() -> list[dict]:
        cameras: list[dict] = LinuxCamerasManager.get_devices_list()
        for camera in cameras:
            camera['Details'] = LinuxCamerasManager.get_camera_details(camera['Device'])
        cameras = [
            camera
            for camera in cameras
            if 'WEBCAM' in camera['Name'].upper()
        ]
        return cameras

    @staticmethod
    @time_me(debug= False)
    def get_devices_list() -> list[dict]:
        cameras: list[dict] = []
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output= True,
            text= True
        )
        if result.returncode != 0:
            msg: str = '"vl4l2-ctl --list-devices" failed.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        device_info: list[str] = [
            dev
            for dev in result.stdout.split('\n\n')
            if dev.strip()
        ]
        for dev in device_info:
            lines: list[str] = [
                line.strip()
                for line in dev.split('\n')
                if line.strip()
            ]
            if not lines:
                continue
            camera_name: str = lines[0][:-1]
            camera_paths: list[str] = [
                line
                for line in lines[1:]
                if line.startswith('/dev/video')
            ]
            for path in camera_paths:
                cameras.append({
                    'Name': camera_name,
                    'Device': path,
                })
        return cameras

    @staticmethod
    @time_me(debug= False)
    def get_camera_details(path: str) -> dict:
        camera_details: dict = {}
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ['v4l2-ctl', f'--device={path}', '-D'],
            capture_output= True,
            text= True
        )
        if result.returncode != 0:
            msg: str = f'"v4l2-ctl --device={path} --all" failed.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        lines: list[str] = [
            line
            for line in result.stdout.split('\n')
            if line.strip()
        ]
        if not lines:
            msg: str = f'"v4l2-ctl --device={path} --all" failed.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)
        for line in lines:
            if line.endswith(':'):
                section: str = line[:-1]
                camera_details[section] = {}
                continue
            if ':' in line:
                key: str
                val: str
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()
                if section: # type: ignore
                    camera_details[section][key] = val
                else:
                    camera_details[key] = val
        return camera_details


def camera_manager_factory(cameraID: Optional[int] = None) -> CameraManager:
    system: str = platform.system().lower()
    if system == 'windows':
        my_logger.info('Windows OS detected.')
        return WindowsCameraManager(cameraID)
    if system == 'linux':
        my_logger.info('Linux OS detected.')
        return LinuxCamerasManager(cameraID)
    raise SystemError('System not supported.')
