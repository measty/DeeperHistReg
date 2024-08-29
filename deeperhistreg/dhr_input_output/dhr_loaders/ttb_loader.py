### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable
import logging
from tiatoolbox.wsicore.wsireader import WSIReader

### External Imports ###
import numpy as np
import torch as tc
import cv2

import pyvips

### Internal Imports ###

from loader import WSILoader, LoadMode
from dhr_utils import utils as u

########################

def smooth_and_resize(image: np.ndarray, sigma: float, resample_ratio: float) -> np.ndarray:
    # Ensure the image is in float32 format
    image = image.astype(np.float32)

    # Apply Gaussian blur
    # Note: cv2.GaussianBlur uses kernel size instead of sigma directly
    # We'll calculate kernel size based on sigma
    kernel_size = int(6 * sigma + 1)  # Ensure it's odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # Resize the image
    # OpenCV's resize function is equivalent to 'linear' interpolation in pyvips
    height, width = smoothed_image.shape[:2]
    new_height = int(height * resample_ratio)
    new_width = int(width * resample_ratio)
    resampled_image = cv2.resize(smoothed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resampled_image



class TTBSlideLoader(WSILoader):
    """
    TODO - documentation
    """
    def __init__(
        self,
        image_path,
        mode=LoadMode.NUMPY):
        """
        
        """
        self.image_path = image_path
        self.mode = mode
        self.image = WSIReader.open(image_path)
        self.num_levels = self.get_num_levels()
        self.resolutions = self.get_resolutions()
        self.bands = 3 # fix me
        
    def get_num_levels(self) -> int:
        """
        TODO - documentation
        """       
        return int(self.image.info.level_count)
    
    def get_resolutions(self) -> Iterable[int]:
        """
        TODO - documentation
        """
        return self.image.info.level_dimensions

    def load(self) -> pyvips.Image:
        """
        TODO - documentation
        """
        return self.image

    def get_best_level(self, resample_ratio : float) -> pyvips.Image:
        """
        TODO - documentation
        """
        resolution = self.image.info.mpp / resample_ratio
        read_level, post_read_scale_factor = self.image._find_optimal_level_and_downsample(
            resolution,
            "mpp",
        )
        return read_level, post_read_scale_factor

    def update_resample_ratio(self, resample_ratio : float, level_to_use : int) -> float:
        """
        TODO - documentation
        """
        original_resolution = self.resolutions[0]
        updated_resolution = self.resolutions[level_to_use]
        resample_ratio =  (original_resolution[0] * resample_ratio) / updated_resolution[0]
        return resample_ratio
    
    def resample_ratio2mpp(self, resample_ratio : float) -> float:
        """
        TODO - documentation
        """
        return self.image.mpp / resample_ratio

    def resample(self, resample_ratio : float) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        level, resample_ratio = self.get_best_level(resample_ratio)
        sigma = u.calculate_smoothing_sigma(resample_ratio.mean())
        image = self.image.read_rect((0, 0), self.resolutions[level], level, coord_space='resolution')
        #smoothed_image = image.gaussblur(sigma)
        #resampled_image = smoothed_image.resize(resample_ratio, kernel='linear', vscale=resample_ratio)
        resampled_image = smooth_and_resize(image, sigma, resample_ratio.mean())
        if self.mode == LoadMode.NUMPY:
            array = resampled_image
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(resampled_image)
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_region(self, level, offset, shape) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        
        #region = image.crop(offset[1], offset[0], shape[1], shape[0])
        region = self.image.read_rect((offset[0], offset[1]), (shape[0], shape[1]), level, coord_space='resolution')
        if self.mode == LoadMode.NUMPY:
            array = region
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(region)
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_regions(self, level, offsets, shape) -> Union[np.ndarray, tc.Tensor]:
        """
        TODO - documentation
        """
        if level >= self.num_levels:
            level = self.num_levels - 1
            logging.warn(f"Only {self.num_levels} are available. Setting level to {self.num_levels - 1}.")
        #image = pyvips.Image.openslideload(self.image_path, level=level)
        array = np.zeros((len(offsets), *shape, self.bands), dtype=np.uint8)
        for i, offset in enumerate(offsets):
            crop = self.image.read_rect((offset[0], offset[1]), (shape[0], shape[1]), level, coord_space='resolution')
            array[i, :, :, :] = crop
        if self.mode == LoadMode.NUMPY:
            pass
        elif self.mode == LoadMode.PYTORCH:
            array = tc.from_numpy(array)
        else:
            raise ValueError("Unsupported mode.")
        return array

    def load_level(self, level) -> Union[np.ndarray, tc.Tensor, pyvips.Image]:
        """
        TODO - documentation
        """
        image = self.image.read_rect((0, 0), self.resolutions[level], level, coord_space='resolution')
        if self.mode == LoadMode.NUMPY:
            array = image
        elif self.mode == LoadMode.PYTORCH:
            array = u.image_to_tensor(image)
        elif self.mode == LoadMode.PYVIPS:
            array = pyvips.Image.new_from_array(image)[0:3]
        else:
            raise ValueError("Unsupported mode.")
        return array

    