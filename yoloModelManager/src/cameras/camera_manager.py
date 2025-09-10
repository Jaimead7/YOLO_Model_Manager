import atexit
import platform
import subprocess
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint
from sys import exit
from time import sleep
from typing import Any, Callable, Generator, Optional, TypedDict

import cv2
import numpy as np
from pyUtils import time_me

from ..filesystem.files import create_dataset_medatada_yaml, save_image
from ..image.image_processing import ImageProcessing
from ..model.model_manager import ModelManager
from ..utils.config import IMAGES_PATH, MY_CFG, my_logger
from ..utils.data_types import DatasetMetadataDict


class CameraInfo(TypedDict):
    index: int
    name: str
    width: int
    height: int
    brightness: float
    contrast: float
    saturation: float
    exposure: float
    wb: float


class CameraManager(ABC):
    def __init__(
        self,
        camera_id: Optional[int] = None
    ) -> None:
        atexit.register(self.cleanup)
        self.camera_info: CameraInfo = self.select_camera(camera_id)
        with self.get_video_capture() as cap:
            self.get_camera_resolution(cap)
        self.show_filters = None
        self.save_filters = None
        self.save_dir_path = None
        self.keys_callbacks: dict[int, tuple[Callable, dict]] = {}
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
    def _brightness(self) -> float:
        return self.camera_info['brightness']

    @_brightness.setter
    def _brightness(self, b: float) -> None:
        self.camera_info['brightness'] = b

    @property
    def _contrast(self) -> float:
        return self.camera_info['contrast']

    @_contrast.setter
    def _contrast(self, b: float) -> None:
        self.camera_info['contrast'] = b

    @property
    def _saturation(self) -> float:
        return self.camera_info['saturation']

    @_saturation.setter
    def _saturation(self, b: float) -> None:
        self.camera_info['saturation'] = b

    @property
    def _exposure(self) -> float:
        return self.camera_info['exposure']

    @_exposure.setter
    def _exposure(self, b: float) -> None:
        self.camera_info['exposure'] = b

    @property
    def _wb(self) -> float:
        return self.camera_info['wb']

    @_wb.setter
    def _wb(self, b: float) -> None:
        self.camera_info['wb'] = b

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
                height= 0,
                brightness= MY_CFG.camera.brightness,
                contrast= MY_CFG.camera.contrast,
                saturation= MY_CFG.camera.saturation,
                exposure= MY_CFG.camera.exposure,
                wb= MY_CFG.camera.wb
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
                msg: str = f'No camera for index {camera_id} was found.'
                my_logger.error(msg)
                raise ValueError(msg)
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
        self._cap: cv2.VideoCapture = cv2.VideoCapture(self.camera)
        if not self._cap.isOpened():
            msg: str = 'Can\'t connect to the camera.'
            my_logger.error(f'ConnectionRefusedError: {msg}')
            raise ConnectionRefusedError(msg)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            yield self._cap
        finally:
            self._cap.release()

    def cleanup(self) -> None:
        if self._cap is not None:
            self._cap.release()
            my_logger.debug(f'Camera {self.name} cleared.')

    @time_me
    def get_camera_resolution(
        self,
        cap: cv2.VideoCapture
    ) -> tuple[int, int]:
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        my_logger.info(f'Camera resolution: {self.width}x{self.height} px.')
        return (self.width, self.height)

    @time_me
    def set_camera_resolution(
        self,
        cap: cv2.VideoCapture
    ) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.get_camera_resolution(cap)

    @time_me
    def get_brightness(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._brightness = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        my_logger.info(f'Camera brightness: {self._brightness}.')
        return self._brightness

    @time_me
    def set_brightness(
        self,
        cap: cv2.VideoCapture,
        b: Optional[float] = None
    ) -> None:
        if b is None:
            b = self._brightness
        cap.set(cv2.CAP_PROP_BRIGHTNESS, b)
        self.get_brightness(cap)

    @time_me
    def get_contrast(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._contrast = int(cap.get(cv2.CAP_PROP_CONTRAST))
        my_logger.info(f'Camera contrast: {self._contrast}.')
        return self._contrast

    @time_me
    def set_contrast(
        self,
        cap: cv2.VideoCapture,
        c: Optional[float] = None
    ) -> None:
        if c is None:
            c = self._contrast
        cap.set(cv2.CAP_PROP_CONTRAST, c)
        self.get_contrast(cap)

    @time_me
    def get_saturation(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._saturation = int(cap.get(cv2.CAP_PROP_SATURATION))
        my_logger.info(f'Camera saturation: {self._saturation}.')
        return self._saturation

    @time_me
    def set_saturation(
        self,
        cap: cv2.VideoCapture,
        s: Optional[float] = None
    ) -> None:
        if s is None:
            s = self._saturation
        cap.set(cv2.CAP_PROP_SATURATION, s)
        self.get_saturation(cap)

    @time_me
    def get_exposure(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._exposure = int(cap.get(cv2.CAP_PROP_EXPOSURE))
        my_logger.info(f'Camera exposure: {self._exposure}.')
        return self._exposure

    @time_me
    def set_exposure(
        self,
        cap: cv2.VideoCapture,
        e: Optional[float] = None
    ) -> None:
        if e is None:
            e = self._exposure
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        self.get_exposure(cap)

    @time_me
    def get_auto_exposure(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._auto_exposure = int(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        my_logger.info(f'Camera auto-exposure: {self._auto_exposure}.')
        return self._auto_exposure

    @time_me
    def set_auto_exposure(
        self,
        cap: cv2.VideoCapture,
        value: Optional[float] = None
    ) -> None:
        if value is None:
            value = self._auto_exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
        self.get_auto_exposure(cap)

    @time_me
    def get_wb(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._wb = int(cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
        my_logger.info(f'Camera wb: {self._wb}.')
        return self._wb

    @time_me
    def set_wb(
        self,
        cap: cv2.VideoCapture,
        t: Optional[float] = None
    ) -> None:
        if t is None:
            t = self._wb
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, t)
        self.get_wb(cap)

    @time_me
    def get_auto_wb(
        self,
        cap: cv2.VideoCapture
    ) -> float:
        self._auto_wb = int(cap.get(cv2.CAP_PROP_AUTO_WB))
        my_logger.info(f'Camera auto-wb: {self._auto_wb}.')
        return self._auto_wb

    @time_me
    def set_auto_wb(
        self,
        cap: cv2.VideoCapture,
        value: Optional[float] = None
    ) -> None:
        if value is None:
            value = self._auto_wb
        cap.set(cv2.CAP_PROP_AUTO_WB, value)
        self.get_wb(cap)

    @time_me
    def reset_window_to_camera_resolution(self) -> None:
        cv2.resizeWindow(self.name, self.width, self.height)
        my_logger.debug(f'Window resize to: {self.width}x{self.height} px.')

    def capture_frame(
        self,
        cap: cv2.VideoCapture
    ) -> None:
        ret: bool
        self.last_frame: np.ndarray
        for _ in range(2):
            cap.grab()
        ret, self.last_frame = cap.retrieve()
        if not ret:
            msg: str = 'Can\'t read frame.'
            my_logger.error(f'RuntimeError: {msg}')
            raise RuntimeError(msg)

    def exit(self, *args, **kwargs) -> int:
        my_logger.info('Stopping stream...')
        data: DatasetMetadataDict = {
            'date': datetime.now(timezone.utc),
            'camera_width': self.width,
            'camera_height': self.height,
            'filters': [ImageProcessing.get_filter_name(filter)
                        for filter in self.save_filters],
            'brightness': self._brightness,
            'contrast': self._contrast,
            'saturation': self._saturation,
            'exposure': self._exposure,
            'wb': self._wb
        }
        create_dataset_medatada_yaml(
            self.save_dir_path,
            data
        )
        return -1

    def save_last_frame(self, *args, **kwargs) -> int:
        try:
            subfolder = Path(kwargs['subfolder'])
        except:
            subfolder = Path("")
        if len(self.save_filters) == 0:
            save_image(self.last_frame, self.save_dir_path / subfolder)
        else:
            frame: np.ndarray = self.last_frame
            for filter in self.save_filters:
                frame: np.ndarray = filter(frame)
            save_image(frame, self.save_dir_path / subfolder)
        return 0

    def load_params_from_model(self, model: ModelManager) -> None:
        self.show_filters = [model.process_frame]
        self.save_filters = model.filters
        self.width = model.camera_width
        self.height = model.camera_height
        self._brightness = model.camera_brightness
        self._contrast = model.camera_contrast
        self._saturation = model.camera_saturation
        self._exposure = model.camera_exposure
        self._wb = model.camera_wb

    def add_cam_prop_bars(
        self,
        cap: cv2.VideoCapture
    ) -> None:
        cv2.createTrackbar(
            'Brightness',
            self.name,
            int(self._brightness),
            255,
            lambda x: self.set_brightness(cap, x)
        )
        cv2.createTrackbar(
            'Contrast',
            self.name,
            int(self._contrast),
            255,
            lambda x: self.set_contrast(cap, x)
        )
        cv2.createTrackbar(
            'Saturation',
            self.name,
            int(self._saturation),
            255,
            lambda x: self.set_saturation(cap, x)
        )
        cv2.createTrackbar(
            'Exposure',
            self.name,
            int(self._exposure),
            255,
            lambda x: self.set_exposure(cap, x)
        )
        cv2.createTrackbar(
            'Temperature',
            self.name,
            int(self._wb),
            255,
            lambda x: self.set_wb(cap, x)
        )

    def video_stream(self) -> None:
        self.keys_callbacks[27] = (self.exit, {})
        cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
        with self.get_video_capture() as cap:
            self.set_auto_exposure(cap, MY_CFG.camera.auto_exposure)
            self.set_auto_wb(cap, MY_CFG.camera.auto_wb)
            self.add_cam_prop_bars(cap)
            self.set_camera_resolution(cap)
            self.reset_window_to_camera_resolution()
            my_logger.debug('Starting video stream.')
            while True:
                self.capture_frame(cap)
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
                        #DELETE: prints
                        print(f'CAP_PROP_BRIGHTNESS -> {cap.get(cv2.CAP_PROP_BRIGHTNESS)}')
                        print(f'CAP_PROP_CONTRAST -> {cap.get(cv2.CAP_PROP_CONTRAST)}')
                        print(f'CAP_PROP_SATURATION -> {cap.get(cv2.CAP_PROP_SATURATION)}')
                        print(f'CAP_PROP_AUTO_EXPOSURE -> {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}')
                        print(f'CAP_PROP_EXPOSURE -> {cap.get(cv2.CAP_PROP_EXPOSURE)}')
                        print(f'CAP_PROP_WB_TEMPERATURE -> {cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}')
                        print(f'CAP_PROP_AUTO_WB -> {cap.get(cv2.CAP_PROP_AUTO_WB)}')
                        my_logger.debug(f'Key pressed: "{key}".')
                sleep(0.2) #DELETE:
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
            msg: str = '"v4l2-ctl --list-devices" failed.'
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
