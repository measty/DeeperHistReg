### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Iterable, Sequence, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Optional

### External Imports ###
import math
import numpy as np
import scipy.ndimage as nd
import pyvips
import torch as tc
import torch.nn.functional as F
import torchvision.transforms as tr
from scipy.ndimage import median_filter, label, binary_dilation
from scipy.sparse import diags, identity
from scipy.sparse.linalg import cg


### Internal Imports ###



def ring_median(img, r_out, r_in=0, mode="reflect"):
    """
    Median of pixels whose distance from the centre lies in (r_in, r_out].

    Parameters
    ----------
    img   : 2-D array
    r_out : int        outer radius  (≥1)
    r_in  : int        inner radius  (≥0, < r_out)
    mode  : str        ndimage border mode

    Returns
    -------
    med   : 2-D array  same shape as img
    """
    if not (0 <= r_in < r_out):
        raise ValueError("Require 0 ≤ r_in < r_out")

    # build an (2*r_out+1)² binary mask with True in the annulus
    y, x = np.ogrid[-r_out:r_out + 1, -r_out:r_out + 1]
    d2 = x * x + y * y
    mask = (d2 <= r_out * r_out) & (d2 > r_in * r_in)  # annulus

    return median_filter(img, footprint=mask, mode=mode)

def sanitize_displacement(field, *, k=1000.0, max_region=50_000,
                           window=25, dilate=17,
                           ds=16,               # <-- new: down-sample factor (int ≥ 1)
                           tol=1e-5, maxiter=2000):
    """
    Removes small discontinuity blobs from a 2xHxW displacement field
    using robust outlier detection on a down-sampled grid + Laplacian in-painting.

    Parameters
    ----------
    field : np.ndarray, shape (2, H, W)
    k : float
        Outlier threshold in MAD units.
    max_region : int
        Max blob size (pixels @ full-res) to in-paint.
    window : int
        Median-filter size @ full-res (must be odd).  Internally shrunk to window//ds.
    dilate : int
        Extra dilation radius (pixels @ full-res).
    ds : int
        Spatial down-sample factor for mask creation (1 ⇒ original resolution).
    tol, maxiter : float, int
        Conjugate-gradient solver settings.

    Returns
    -------
    clean : np.ndarray, same shape as `field`
    """
    if ds < 1 or not isinstance(ds, int):
        raise ValueError("ds must be a positive integer")
    u0, v0 = field[0,:,:], field[1,:,:]
    H, W = u0.shape

    # Down-sample for *mask* computation   (nearest-neighbour)
    print("getting mask")
    if ds > 1:
        u_ds = u0[::ds, ::ds]
        v_ds = v0[::ds, ::ds]
        win_ds = window # max(3, (window // ds) | 1)        # ensure odd and ≥3
    else:
        u_ds, v_ds, win_ds = u0, v0, window

    # robust local median / MAD on coarse grid

    # Choose radii (full-res) then scale to coarse grid
    r_out = window         
    r_in  = int(window /2)                   # ignore centre ± 1 px
    #r_out = max(1, r_out_full // ds)
    #r_in  = max(0, r_in_full  // ds)

    u_med = ring_median(u_ds, r_out, r_in)
    v_med = ring_median(v_ds, r_out, r_in)
    #u_med = median_filter(u_ds, size=win_ds, mode='reflect')
    #v_med = median_filter(v_ds, size=win_ds, mode='reflect')
    #mad_u = median_filter(np.abs(u_ds - u_med), size=win_ds, mode='reflect')
    #mad_v = median_filter(np.abs(v_ds - v_med), size=win_ds, mode='reflect')
    mad = 1 # mad_u + mad_v + 1e-12

    outlier_ds = ((np.abs(u_ds - u_med) + np.abs(v_ds - v_med)) / mad) > k

    # keep small connected components
    structure = np.ones((3, 3), dtype=bool)         # 8-connectivity
    lab, nlab = label(outlier_ds, structure)
    mask_ds = np.zeros_like(outlier_ds, dtype=bool)
    px_per_blob_limit = max_region # // (ds * ds)
    for n in range(1, nlab + 1):
        if (lab == n).sum() <= px_per_blob_limit:
            mask_ds[lab == n] = True

    # upsample mask back to full resolution (nearest-neighbour)
    if ds > 1:
        mask_full = np.kron(mask_ds, np.ones((ds, ds), dtype=bool))
        mask_full = mask_full[:H, :W]               # trim edge padding
    else:
        mask_full = mask_ds

    # optional dilation on full-res mask
    if dilate > 0:
        mask_full = binary_dilation(mask_full, iterations=dilate)

    if not mask_full.any():
        return field.copy()

    # Poisson in-painting
    print("inpainting")
    clean = field.copy()
    N = H * W
    main = np.ones(N) * 4
    off1 = np.ones(N - 1) * -1
    off1[np.arange(1, N) % W == 0] = 0
    offW = np.ones(N - W) * -1
    L = diags([main, off1, off1, offW, offW],
              [0, -1, +1, -W, +W], format='csr')

    for c in range(2):
        vec = field[c, :,:].ravel()
        known = ~mask_full.ravel()
        A = L[mask_full.ravel()][:, mask_full.ravel()]
        b = -L[mask_full.ravel()][:, known] @ vec[known]
        x, info = cg(A, b, rtol=tol, maxiter=maxiter)
        if info != 0:
            print(f"Warning: CG did not converge (info={info})")
        vec_out = vec.copy()
        vec_out[mask_full.ravel()] = x
        clean[c, :,:] = vec_out.reshape(H, W)

    return clean

def save_ncc_heatmap_image(
    ncc_map: tc.Tensor,
    save_path: Union[str, Path],
    colormap: str = "viridis",
    slice_idx: Optional[int] = None,
    projection_axis: Optional[int] = None,
    normalize: bool = True,
    invert: bool = False,
    percentile_range: tuple = (2, 96),
    padding_params: dict = None
) -> None:
    """
    Save NCC quality map as a simple PNG heatmap image without axes or labels.
    
    Parameters
    ----------
    ncc_map : tc.Tensor
        The NCC quality map tensor returned from ncc_local function
        Expected shape: (B, 1, H, W) for 2D or (B, 1, D, H, W) for 3D
    save_path : str or Path
        Path where to save the PNG file
    colormap : str, default="viridis"
        Matplotlib colormap name (e.g., 'viridis', 'hot', 'jet', 'plasma', 'coolwarm')
    slice_idx : int, optional
        For 3D data, which slice to visualize (along depth dimension)
        If None, uses middle slice
    projection_axis : int, optional
        For 3D data, axis along which to do max projection
        If provided, overrides slice_idx. 0=depth, 1=height, 2=width
    normalize : bool, default=True
        Whether to normalize values to [0, 1] range before applying colormap
    invert : bool, default=False
        Whether to invert the colormap (useful if higher NCC values should be darker)
    percentile_range : tuple, default=(2, 96)
        Percentile range to use for normalization.
        Values outside this range will be clipped to 0 or 1.
    """

    # Convert to numpy and remove batch/channel dimensions
    if isinstance(ncc_map, tc.Tensor):
        ncc_array = ncc_map.detach().cpu().numpy()
    else:
        ncc_array = ncc_map
    
    # Handle batch dimension - take first batch
    if len(ncc_array.shape) >= 3:
        ncc_array = ncc_array[0]  # Remove batch dimension
    
    # Handle channel dimension if present
    if len(ncc_array.shape) >= 3 and ncc_array.shape[0] == 1:
        ncc_array = ncc_array[0]  # Remove channel dimension
    
    # Handle 3D data
    if len(ncc_array.shape) == 3:
        if projection_axis is not None:
            # Max projection along specified axis
            ncc_array = np.max(ncc_array, axis=projection_axis)
        else:
            # Take a slice
            if slice_idx is None:
                slice_idx = ncc_array.shape[0] // 2  # Middle slice
            ncc_array = ncc_array[slice_idx]
    
    # Ensure we have 2D data at this point
    if len(ncc_array.shape) != 2:
        raise ValueError(f"Expected 2D array after processing, got shape {ncc_array.shape}")
    
    # Robust normalization using percentiles to avoid outlier distortion
    if normalize:
        ncc_p_low = np.percentile(ncc_array, percentile_range[0])
        ncc_p_high = np.percentile(ncc_array, percentile_range[1])
        if ncc_p_high > ncc_p_low:
            ncc_array = (ncc_array - ncc_p_low) / (ncc_p_high - ncc_p_low)
            # Clip values outside [0, 1] range
            ncc_array = np.clip(ncc_array, 0, 1)
        else:
            ncc_array = np.zeros_like(ncc_array)
    
    # Invert if requested
    if invert:
        ncc_array = 1.0 - ncc_array
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored_array = cmap(ncc_array)  # Returns RGBA values in [0, 1]
    
    # Convert to 8-bit RGB
    rgb_array = (colored_array[:, :, :3] * 255).astype(np.uint8)
    
    # Convert to PIL Image and save
    image = Image.fromarray(rgb_array)
    image.save(save_path)
    
    print(f"NCC heatmap image saved to: {save_path}")
    print(f"Image size: {image.size}")
    if normalize:
        print(f"Normalized using {percentile_range[0]}th-{percentile_range[1]}th percentile range: [{ncc_p_low:.3f}, {ncc_p_high:.3f}]")


def normalize(tensor : Union[tc.Tensor, np.ndarray]) -> Union[tc.Tensor, np.ndarray]:
    """
    TODO
    """
    if isinstance(tensor, tc.Tensor):
        if len(tensor.size()) - 2 == 2:
            num_channels = tensor.size(1)
            normalized_tensor = tc.zeros_like(tensor)
            for i in range(num_channels):
                mins, _ = tc.min(tc.min(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True) # TODO - find better approach
                maxs, _ = tc.max(tc.max(tensor[:, i, :, :] , dim=1, keepdim=True)[0], dim=2, keepdim=True)
                normalized_tensor[:, i, :, :] = (tensor[:, i, :, :] - mins) / (maxs - mins)
            return normalized_tensor
        else:
            raise ValueError("Unsupported number of channels.")
        
    elif isinstance(tensor, np.ndarray):
        if len(tensor.shape) == 2:
            return (tensor - np.min(tensor)) / (np.max(tensor) - np.min(tensor))
        elif len(tensor.shape) == 3:
            normalized_image = np.zeros_like(tensor)
            for i in range(normalized_image.shape[2]):
                normalized_image[:, :, i] = normalize(tensor[:, :, i])
            return normalized_image
        else:
            raise ValueError("Unsupported number of channels.")
    else:
        raise ValueError("Unsupported array library.")
    
def normalize_to_window(tensor: Union[tc.Tensor, np.ndarray], min_value : float, max_value : float) -> Union[tc.Tensor, np.ndarray]:
    """
    TODO
    """
    return normalize(tensor) * (max_value - min_value) + min_value

def resample(tensor : tc.Tensor, resample_ratio : float, mode: str="bilinear") -> tc.Tensor:
    """
    TODO
    """
    return F.interpolate(tensor, scale_factor = 1 / resample_ratio, mode=mode, recompute_scale_factor=False, align_corners=False)

def resample_tensor_to_size(tensor: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear') -> tc.Tensor:
    """
    TODO
    """
    return F.interpolate(tensor, size=new_size, mode=mode, align_corners=False)

def resample_displacement_field(displacement_field : tc.Tensor, resample_ratio : float, mode: str="bilinear") -> tc.Tensor:
    """
    TODO
    """
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), scale_factor = 1 / resample_ratio, mode=mode, recompute_scale_factor=False, align_corners=False).permute(0, 2, 3, 1)

def resample_displacement_field_to_size(displacement_field: tc.Tensor, new_size: tc.Tensor, mode: str='bilinear') -> tc.Tensor:
    """
    TODO
    """
    return F.interpolate(displacement_field.permute(0, 3, 1, 2), size=new_size, mode=mode, align_corners=False).permute(0, 2, 3, 1)

def gaussian_smoothing(tensor : tc.Tensor, sigma : float) -> tc.Tensor:
    """
    TODO
    """
    with tc.set_grad_enabled(False):
        kernel_size = int(sigma * 2.54) + 1 if int(sigma * 2.54) % 2 == 0 else int(sigma * 2.54)
        return tr.GaussianBlur(kernel_size, sigma)(tensor)

def gaussian_smoothing_np(image : np.ndarray, sigma : float) -> np.ndarray:
    """
    TODO
    """
    output = image.copy()
    for i in range(image.shape[2]):
        output[:, :, i] = nd.gaussian_filter(image[:, :, i], sigma)
    return output

def gaussian_smoothing_patch(
    tensor : tc.Tensor,
    sigma : float,
    patch_size : tuple=(2048, 2048),
    offset : tuple=(50, 50)) -> tc.Tensor:
    """
    TODO
    """
    smoothed_tensor = tc.zeros_like(tensor)
    with tc.set_grad_enabled(False):
        y_size, x_size = tensor.size(2), tensor.size(3)
        rows, cols = int(np.ceil(y_size / patch_size[0])), int(np.ceil(x_size / patch_size[1]))
        for row in range(rows):
            for col in range(cols):
                b_x = max(0, min(x_size, col*patch_size[1]))
                b_y = max(0, min(y_size, row*patch_size[0]))
                e_x = max(0, min(x_size, (col+1)*patch_size[1]))
                e_y = max(0, min(y_size, (row+1)*patch_size[0]))
                ob_x = max(0, min(x_size, b_x - offset[1]))
                oe_x = max(0, min(x_size, e_x + offset[1]))
                ob_y = max(0, min(y_size, b_y - offset[0]))
                oe_y =  max(0, min(y_size, e_y + offset[0]))
                diff_bx = b_x - ob_x
                diff_by = b_y - ob_y
                smoothed_tensor[:, :, b_y:e_y, b_x:e_x] = gaussian_smoothing(tensor[:, :, ob_y:oe_y, ob_x:oe_x], sigma)[:, :, diff_by:diff_by+patch_size[0], diff_bx:diff_bx+patch_size[1]]
    return smoothed_tensor

def get_combined_size(tensor_1 : tc.Tensor, tensor_2 : tc.Tensor) -> Iterable[int]:
    """
    TODO
    """
    tensor_1_y_size, tensor_1_x_size = tensor_1.size(2), tensor_1.size(3)
    tensor_2_y_size, tensor_2_x_size = tensor_2.size(2), tensor_2.size(3)
    return tensor_1_y_size, tensor_1_x_size, tensor_2_y_size, tensor_2_x_size

def create_identity_displacement_field(tensor : tc.Tensor) -> tc.Tensor:
    """
    TODO
    """
    return tc.zeros((tensor.size(0), tensor.size(2), tensor.size(3)) + (2,)).type_as(tensor)

def create_identity_transform(tensor : tc.Tensor) -> tc.Tensor:
    """
    TODO
    """
    transform = tc.zeros((tensor.size(0), 2, 3)).type_as(tensor)
    transform[:, 0, 0] = 1
    transform[:, 1, 1] = 1
    return transform


def calculate_pad_value(size_1 : Iterable[int], size_2 : Iterable[int]) -> Tuple[Iterable[tuple], Iterable[tuple]]:
    """
    TODO
    """
    y_size_1, x_size_1 = size_1
    y_size_2, x_size_2 = size_2
    pad_1 = [(0, 0), (0, 0)]
    pad_2 = [(0, 0), (0, 0)]
    if y_size_1 > y_size_2:
        pad_size = y_size_1 - y_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[0] = pad
    elif y_size_1 < y_size_2:
        pad_size = y_size_2 - y_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[0] = pad
    else:
        pass
    if x_size_1 > x_size_2:
        pad_size = x_size_1 - x_size_2
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_2[1] = pad
    elif x_size_1 < x_size_2:
        pad_size = x_size_2 - x_size_1
        pad = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        pad_1[1] = pad
    else:
        pass
    return pad_1, pad_2

def pad_to_same_size(
    image_1 : Union[tc.Tensor, np.ndarray, pyvips.Image],
    image_2 : Union[tc.Tensor, np.ndarray, pyvips.Image],
    pad_value : float=1.0) -> Tuple[Union[tc.Tensor, np.ndarray, pyvips.Image], Union[tc.Tensor, np.ndarray, pyvips.Image], dict]:
    """
    TODO
    """
    if all([isinstance(image, np.ndarray) for image in [image_1, image_2]]):
        return pad_to_same_size_np(image_1, image_2, pad_value)
    elif all([isinstance(image, tc.Tensor) for image in [image_1, image_2]]):
        return pad_to_same_size_tc(image_1, image_2, pad_value)
    elif all([isinstance(image, pyvips.Image) for image in [image_1, image_2]]):
        return pad_to_same_size_pyvips(image_1, image_2, pad_value)
    else:
        raise ValueError("Unsupported type.")

def pad_to_same_size_np(image_1 : np.ndarray, image_2 : np.ndarray, pad_value : float=1.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    TODO
    """
    y_size_1, x_size_1 = image_1.shape[0], image_1.shape[1]
    y_size_2, x_size_2 = image_2.shape[0], image_2.shape[1]
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    image_1 = np.pad(image_1, ((pad_1[0][0], pad_1[0][1]), (pad_1[1][0], pad_1[1][1]), (0, 0)), mode='constant', constant_values=pad_value)
    image_2 = np.pad(image_2, ((pad_2[0][0], pad_2[0][1]), (pad_2[1][0], pad_2[1][1]), (0, 0)), mode='constant', constant_values=pad_value)
    padding_params = dict()
    padding_params['pad_1'] = pad_1
    padding_params['pad_2'] = pad_2
    return image_1, image_2, padding_params


def pad_to_same_size_pyvips(image_1 : pyvips.Image, image_2 : pyvips.Image, pad_value : float=1.0) -> Tuple[pyvips.Image, pyvips.Image, dict]:
    """
    TODO
    """
    y_size_1, x_size_1 = image_1.height, image_1.width
    y_size_2, x_size_2 = image_2.height, image_2.width
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    image_1 = image_1.gravity("centre", x_size_1 + pad_1[1][0] + pad_1[1][1], y_size_1 + pad_1[0][0] + pad_1[0][1], background=[pad_value, pad_value, pad_value])
    image_2 = image_2.gravity("centre", x_size_2 + pad_2[1][0] + pad_2[1][1], y_size_2 + pad_2[0][0] + pad_2[0][1], background=[pad_value, pad_value, pad_value])
    padding_params = dict()
    padding_params['pad_1'] = pad_1
    padding_params['pad_2'] = pad_2
    return image_1, image_2, padding_params

def pad_to_same_size_tc(image_1 : tc.Tensor, image_2 : tc.Tensor, pad_value : float=1.0) -> Tuple[tc.Tensor, tc.Tensor, dict]:
    """
    TODO
    """
    to_revert = False
    if len(image_1.shape) == 4 and len(image_2.shape) == 4:
        pass
    elif len(image_1.shape) == 3 and len(image_2.shape) == 3:
        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)
        to_revert = True
    else:
        raise ValueError("Unsupported size.")
    y_size_1, x_size_1 = image_1.size(2), image_1.size(3)
    y_size_2, x_size_2 = image_2.size(2), image_2.size(3)
    pad_1, pad_2 = calculate_pad_value((y_size_1, x_size_1), (y_size_2, x_size_2))
    image_1 = F.pad(image_1, pad_1[1] + pad_1[0], mode='constant', value=pad_value)
    image_2 = F.pad(image_2, pad_2[1] + pad_2[0], mode='constant', value=pad_value)
    padding_params = dict()
    padding_params['pad_1'] = pad_1
    padding_params['pad_2'] = pad_2
    if to_revert:
        image_1 = image_1.squeeze(0)
        image_2 = image_2.squeeze(0)
    return image_1, image_2, padding_params


def unpad(
    image_1 : Union[tc.Tensor, np.ndarray, pyvips.Image],
    image_2 : Union[tc.Tensor, np.ndarray, pyvips.Image],
    padding_params : dict,
    unpad_with_target : bool = False) -> Tuple[Union[tc.Tensor, np.ndarray, pyvips.Image], Union[tc.Tensor, np.ndarray, pyvips.Image]]:
    """
    TODO
    """
    if all([isinstance(image, np.ndarray) for image in [image_1, image_2]]):
        return unpad_np(image_1, image_2, padding_params, unpad_with_target)
    elif all([isinstance(image, tc.Tensor) for image in [image_1, image_2]]):
        return unpad_tc(image_1, image_2, padding_params, unpad_with_target)
    elif all([isinstance(image, pyvips.Image) for image in [image_1, image_2]]):
        return unpad_pyvips(image_1, image_2, padding_params, unpad_with_target)
    else:
        raise ValueError("Unsupported type.")
    
def unpad_np(
    image_1 : np.ndarray,
    image_2 : np.ndarray,
    padding_params : dict,
    unpad_with_target : bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO
    """
    sp = padding_params['pad_2'] if unpad_with_target else padding_params['pad_1']
    tp = padding_params['pad_2']
    image_1 = image_1[sp[0][0]:image_1.shape[0]-sp[0][1], sp[1][0]:image_1.shape[1]-sp[1][1], :]
    image_2 = image_2[tp[0][0]:image_2.shape[0]-tp[0][1], tp[1][0]:image_2.shape[1]-tp[1][1], :]
    return image_1, image_2
    
def unpad_tc(
    image_1 : tc.Tensor,
    image_2 : tc.Tensor,
    padding_params : dict,
    unpad_with_target : bool) -> Tuple[tc.Tensor, tc.Tensor]:
    """
    TODO
    """
    sp = padding_params['pad_2'] if unpad_with_target else padding_params['pad_1']
    tp = padding_params['pad_2']
    image_1 = image_1[:, :, sp[0][0]:image_1.shape[2]-sp[0][1], sp[1][0]:image_1.shape[3]-sp[1][1]]
    image_2 = image_2[:, :, tp[0][0]:image_2.shape[2]-tp[0][1], tp[1][0]:image_2.shape[3]-tp[1][1]]
    return image_1, image_2

def unpad_pyvips(
    image_1 : pyvips.Image,
    image_2 : pyvips.Image,
    padding_params : dict,
    unpad_with_target : bool) -> Tuple[pyvips.Image, pyvips.Image]:
    """
    TODO
    """
    sp = padding_params['pad_2'] if unpad_with_target else padding_params['pad_1']
    tp = padding_params['pad_2']
    image_1 = image_1.crop(sp[1][0], sp[0][0], image_1.width - sp[1][1] - sp[1][0], image_1.height - sp[0][1] - sp[0][0])
    image_2 = image_2.crop(tp[1][0], tp[0][0], image_2.width - tp[1][1] - tp[1][0], image_2.height - tp[0][1] - tp[0][0])
    return image_1, image_2

def crop_to_template(
    image : Union[tc.Tensor, np.ndarray, pyvips.Image],
    template : Union[tc.Tensor, np.ndarray, pyvips.Image],
    ) -> Union[tc.Tensor, np.ndarray, pyvips.Image]:
    """
    TODO
    """
    if all([isinstance(image, np.ndarray) for image in [image, template]]):
        return crop_to_template_np(image, template)
    elif all([isinstance(image, tc.Tensor) for image in [image, template]]):
        return crop_to_template_tc(image, template)
    elif all([isinstance(image, pyvips.Image) for image in [image, template]]):
        return crop_to_template_pyvips(image, template)
    else:
        raise ValueError("Unsupported type.")
    
def crop_to_template_np(
    image : np.ndarray,
    template : np.ndarray,
    ) -> np.ndarray:
    """
    TODO
    """
    return image[:template.shape[0], :template.shape[1], :]

def crop_to_template_tc(
    image : tc.Tensor,
    template : tc.Tensor,
    ) -> tc.Tensor:
    """
    TODO
    """
    return image[:, :, :template.shape[2], :template.shape[3]]

def crop_to_template_pyvips(
    image : pyvips.Image,
    template : pyvips.Image,
    ) -> tc.Tensor:
    """
    TODO
    """
    return image.crop(0, 0, template.width, template.height)

def calculate_diagonal(tensor : tc.Tensor) -> float:
    """
    TODO - documentation
    """
    return math.sqrt(tensor.size(2)**2 + tensor.size(3)**2)

def calculate_diag(width : float, height : float) -> float:
    """
    TODO - documentation
    """
    return math.sqrt(width**2 + height**2)

def calculate_resample_ratio_based_on_diagonal(old_diagonal : float, new_diagonal : float) -> float:
    """
    TODO - documentation
    """
    return math.sqrt(old_diagonal / new_diagonal)

def convert_to_gray(tensor : tc.Tensor) -> tc.Tensor:
    return tr.Grayscale()(tensor)

def unpad_displacement_field(displacement_field : tc.Tensor, padding_params : dict) -> tc.Tensor:
    """
    TODO
    """
    pad = padding_params['pad_1']
    y_pad, x_pad = pad
    if y_pad[1] == 0:
        displacement_field = displacement_field[:, y_pad[0]:, :, :]
    else:
        displacement_field = displacement_field[:, y_pad[0]:-y_pad[1], :, :]
    if x_pad[1] == 0:
        displacement_field = displacement_field[:, :, x_pad[0]:, :]
    else:
        displacement_field = displacement_field[:, :, x_pad[0]:-x_pad[1], :]
    return displacement_field

def center_of_mass(tensor : tc.Tensor) -> Tuple[int, int]:
    """
    TODO
    """
    y_size, x_size = tensor.size(2), tensor.size(3)
    gy, gx = tc.meshgrid(tc.arange(y_size).type_as(tensor), tc.arange(x_size).type_as(tensor), indexing='ij')
    m00 = tc.sum(tensor).item()
    m10 = tc.sum(gx*tensor).item()
    m01 = tc.sum(gy*tensor).item()
    com_x = m10 / m00
    com_y = m01 / m00
    return com_x, com_y

def tensor_gradient(tensor: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
    """
    Galculates the central gradient of order 1 of the input tensor with respect to the coordinates.

    Parameters
    ----------
    tensor : tc.Tensor
        The input tensor
  
    Returns
    ----------
    gradient : tuple of tc.Tensor
        The tuple containing calculated gradients (in the order Y -> X (2-D) -> Z(3-D))
    """
    device = tensor.device
    gfilter_x = tc.Tensor([
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0],
    ], device=device).type(tensor.type())
    gfilter_y = tc.Tensor([
        [0, -1, 0],
        [0, 0, 0],
        [0, 1, 0],
    ], device=device).type(tensor.type())
    gradient_x = F.conv2d(tensor, gfilter_x.view(1, 1, 3, 3), padding=1) / 2.0
    gradient_y = F.conv2d(tensor, gfilter_y.view(1, 1, 3, 3), padding=1) / 2.0
    return gradient_y, gradient_x

def create_pyramid(tensor: tc.Tensor, num_levels: int, mode: str='bilinear') -> Iterable[tc.Tensor]:
    """
    Creates the resolution pyramid of the input tensor (assuming uniform resampling step = 2).

    Parameters
    ----------
    tensor : tc.Tensor
        The input tensor
    num_levels: int
        The number of output levels
    mode : str
        The interpolation mode ("bilinear" or "nearest")
    
    Returns
    ----------
    pyramid: list of tc.Tensor
        The created resolution pyramid

    """
    pyramid = [None]*num_levels
    for i in range(num_levels - 1, -1, -1):
        if i == num_levels - 1:
            pyramid[i] = tensor
        else:
            current_size = pyramid[i+1].size()
            new_size = (int(current_size[j] / 2) if j > 1 else current_size[j] for j in range(len(current_size)))
            new_size = tc.Size(new_size)[2:]
            new_tensor = resample_tensor_to_size(gaussian_smoothing(pyramid[i+1], 1), new_size, mode=mode)
            pyramid[i] = new_tensor
    return pyramid

def tensor_laplacian(tensor : tc.Tensor) -> tc.Tensor:
    """
    TODO
    """
    laplacian_filter = tc.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).type_as(tensor)
    laplacian = F.conv2d(tensor, laplacian_filter.view(1, 1, 3, 3), padding=1) / 5
    return laplacian

def fold(
    unfolded_tensor : tc.Tensor,
    padded_output_size : tuple,
    padding_tuple : tuple,
    patch_size : tuple,
    stride : int,
    overlap=False) -> tc.Tensor:
    """
    TODO
    """
    new_tensor = tc.zeros((1, unfolded_tensor.size(1),) + padded_output_size).type_as(unfolded_tensor)
    col_y, col_x = int(padded_output_size[0] / stride), int(padded_output_size[1] / stride)
    for j in range(col_y):
        for i in range(col_x):
            if overlap:
                current_patch = unfolded_tensor[j*col_x + i, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
                b_x = i*stride + int(stride/2)
                e_x = (i+1)*stride + int(stride/2)
                b_y = j*stride + int(stride/2)
                e_y = (j+1)*stride + int(stride/2)
                new_tensor[0, :, b_y:e_y, b_x:e_x] = current_patch
            else:
                b_x = i * stride
                e_x = (i+1) * stride
                b_y = j * stride
                e_y = (j+1) * stride
                current_patch = unfolded_tensor[j*col_x + i, :, :, :]
                new_tensor[0, :, b_y:e_y, b_x:e_x] = current_patch

    if padding_tuple[2] == 0 and padding_tuple[3] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:, padding_tuple[0]:]
    elif padding_tuple[2] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:-padding_tuple[3], padding_tuple[0]:]
    elif padding_tuple[3] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[1]:, padding_tuple[0]:-padding_tuple[2]]
    else:
        new_tensor = new_tensor[:, :, padding_tuple[1]:-padding_tuple[3], padding_tuple[0]:-padding_tuple[2]]
    return new_tensor

def unfold(tensor : tc.Tensor, patch_size : tuple, stride : int) -> Tuple[tc.Tensor, tuple, tuple]:
    """
    TODO
    """
    unfolder = tc.nn.Unfold(patch_size, stride=stride)
    pad_x = math.ceil(tensor.size(3) / patch_size[1])*patch_size[1] - tensor.size(3)
    pad_y = math.ceil(tensor.size(2) / patch_size[0])*patch_size[0] - tensor.size(2)
    b_x, e_x = math.floor(pad_x / 2) + patch_size[0], math.ceil(pad_x / 2) + patch_size[0]
    b_y, e_y = math.floor(pad_y / 2) + patch_size[1], math.ceil(pad_y / 2) + patch_size[1]
    new_tensor = F.pad(tensor, (b_x, e_x, b_y, e_y))
    padding_tuple = (b_x, b_y, e_x, e_y)
    padded_output_size = (new_tensor.size(2), new_tensor.size(3))
    new_tensor = unfolder(new_tensor)
    new_tensor = new_tensor.view(new_tensor.size(0), tensor.size(1), patch_size[0], patch_size[1], new_tensor.size(2))
    new_tensor = new_tensor[0].permute(3, 0, 1, 2)
    return new_tensor, padded_output_size, padding_tuple

def calculate_resampling_ratio(x_sizes : Iterable, y_sizes : Iterable, min_resolution : int) -> float:
    """
    TODO
    """
    x_size, y_size = max(x_sizes), max(y_sizes)
    min_size = min(x_size, y_size)
    if min_resolution > min_size:
        resampling_ratio = 1
    else:
        resampling_ratio = min_size / min_resolution
    return resampling_ratio

def initial_resampling(
    source : Union[tc.Tensor, np.ndarray],
    target : Union[tc.Tensor, np.ndarray],
    resolution : int) -> Tuple[Union[tc.Tensor, np.ndarray], Union[tc.Tensor, np.ndarray]]:
    """
    TODO
    """
    source_y_size, source_x_size, target_y_size, target_x_size = get_combined_size(source, target)
    resample_ratio = calculate_resampling_ratio((source_x_size, target_x_size), (source_y_size, target_y_size), resolution)
    resampled_source = resample(gaussian_smoothing(source, min(max(resample_ratio -1, 0.1), 10)), resample_ratio)
    resampled_target = resample(gaussian_smoothing(target, min(max(resample_ratio -1, 0.1), 10)), resample_ratio)
    return resampled_source, resampled_target

def calculate_smoothing_sigma(resample_ratio : float) -> float:
    """
    TODO
    """
    sigma = max(1 / resample_ratio - 1, 0.1)
    return sigma

def round_up_to_odd(value : int) -> int:
    """
    TODO
    """
    return int(np.ceil(value) // 2 * 2 + 1)

def calculate_affine_transform(source_points : np.ndarray, target_points : np.ndarray) -> np.ndarray:
    """
    TODO
    """
    transform, _, _, _ = np.linalg.lstsq(source_points, target_points, rcond=None)
    transform = transform.T    
    return transform

def points_to_homogeneous_representation(points: np.ndarray) -> np.ndarray:
    """
    TODO
    """
    homogenous_points = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    return homogenous_points

def calculate_rigid_transform(source_points : np.ndarray, target_points : np.ndarray) -> np.ndarray:
    """
    TODO
    """
    target = target_points.ravel()
    source_homogenous_points = points_to_homogeneous_representation(source_points)
    source = np.zeros((2*source_points.shape[0], 2*source_points.shape[1]))
    source[0::2, 0:target_points.shape[1]+1] = source_homogenous_points[:, :]
    source[1::2, 0:target_points.shape[1]+1] = source_homogenous_points[:, :]
    source[1::2, 1], source[1::2, 0] = (-1) * source.copy()[1::2, 0], source.copy()[1::2, 1]
    source[1::2, 2] = 0
    source[1::2, 3] = 1
    inv_source = np.linalg.pinv(source)
    params = inv_source @ target 
    transform = np.array([
        [params[0], params[1], params[2]],
        [-params[1], params[0], params[3]],
        [0, 0, 1],
    ], dtype=source_points.dtype)
    return transform

def image_to_tensor(image : np.ndarray, device : str="cpu") -> tc.Tensor:
    """
    TODO
    """
    if len(image.shape) == 3:
        return tc.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    elif len(image.shape) == 2:
        return tc.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

def tensor_to_image(tensor : tc.Tensor) -> np.ndarray:
    """
    TODO
    """
    if tensor.size(0) == 1:
        return tensor[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
    else:
        return tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

def pad_landmarks(landmarks : np.ndarray, padding_size : tuple) -> np.ndarray:
    """
    TODO
    """
    y_pad = padding_size[0]
    x_pad = padding_size[1]
    landmarks[:, 0] = landmarks[:, 0] + x_pad[0]
    landmarks[:, 1] = landmarks[:, 1] + y_pad[0]
    return landmarks

def unpad_landmarks(landmarks : np.ndarray, padding_size : tuple) -> np.ndarray:
    """
    TODO
    """
    y_pad = padding_size[0]
    x_pad = padding_size[1]
    landmarks[:, 0] = landmarks[:, 0] - x_pad[0]
    landmarks[:, 1] = landmarks[:, 1] - y_pad[0]
    return landmarks

def np_df_to_tc_df(displacement_field_np: np.ndarray, device: str="cpu") -> tc.Tensor:
    """
    Convert the displacement field in NumPy to the displacement field in PyTorch (assuming uniform spacing and align_corners set to false).

    Parameters
    ----------
    displacement_field_np : np.ndarray
        The NumPy displacment field (DxYxXxZ)

    Returns
    ----------
     displacement_field_tc : tc.Tensor
        The PyTorch displacement field (1xZxXxYxD)
    """
    shape = displacement_field_np.shape
    ndim = len(shape) - 1
    if ndim == 2:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, 0] = temp_df_copy[:, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, 1] = temp_df_copy[:, :, :, 1] / (shape[1]) * 2.0
    if ndim == 3:
        displacement_field_tc = tc.from_numpy(displacement_field_np.copy())
        displacement_field_tc = displacement_field_tc.permute(1, 2, 3, 0).unsqueeze(0)
        temp_df_copy = displacement_field_tc.clone()
        displacement_field_tc[:, :, :, :, 0] = temp_df_copy[:, :, :, :, 2] / (shape[3]) * 2.0
        displacement_field_tc[:, :, :, :, 1] = temp_df_copy[:, :, :, :, 0] / (shape[2]) * 2.0
        displacement_field_tc[:, :, :, :, 2] = temp_df_copy[:, :, :, :, 1] / (shape[1]) * 2.0
    return displacement_field_tc.to(device)

def tc_df_to_np_df(displacement_field_tc: tc.Tensor) -> np.ndarray:
    """
    Convert the displacement field in PyTorch to the displacement field in NumPy (assuming uniform spacing and align_corners set to false).
    Be careful - it does not convert the whole batch of dfs.

    Parameters
    ----------
    displacement_field_tc : tc.Tensor
        The PyTorch displacement field (1xZxXxYxD)

    Returns
    ----------
    displacement_field_np : np.ndarray
        The NumPy displacment field (DxYxXxZ)
    """
    ndim = len(displacement_field_tc.size()) - 2
    if ndim == 2:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(2, 0, 1).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :] = temp_df_copy[0, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :] = temp_df_copy[1, :, :] / 2.0 * (shape[1])
    elif ndim == 3:
        displacement_field_np = displacement_field_tc.detach().clone().cpu()[0].permute(3, 0, 1, 2).numpy()
        shape = displacement_field_np.shape
        temp_df_copy = displacement_field_np.copy()
        displacement_field_np[0, :, :, :] = temp_df_copy[1, :, :, :] / 2.0 * (shape[2])
        displacement_field_np[1, :, :, :] = temp_df_copy[2, :, :, :] / 2.0 * (shape[1])
        displacement_field_np[2, :, :, :] = temp_df_copy[0, :, :, :] / 2.0 * (shape[3])
    return displacement_field_np

def np_df_to_pyvips_df(displacement_field : np.ndarray) -> pyvips.Image:
    """
    TODO
    """
    grid_x, grid_y = np.meshgrid(np.arange(displacement_field.shape[2]), np.arange(displacement_field.shape[1]))
    df = displacement_field.copy()
    df[0, :, :] += grid_x
    df[1, :, :] += grid_y
    df_vips = pyvips.Image.new_from_array(df.swapaxes(0, 1).swapaxes(1, 2))
    return df_vips

def get_extension(file_path : str) -> str:
    """
    TODO
    """
    _, extension = os.path.splitext(file_path)
    return extension

def array_to_pyvips(array : np.ndarray) -> pyvips.Image:
    """
    TODO
    """
    pyvips_image = pyvips.Image.new_from_array(array)
    return pyvips_image


def calculate_tre(source_landmarks : np.ndarray, target_landmarks : np.ndarray, image_diagonal : float = None):
    if image_diagonal is None:
        tre = np.sqrt(np.square(source_landmarks[:, 0] - target_landmarks[:, 0]) + np.square(source_landmarks[:, 1] - target_landmarks[:, 1]))
    else:
        tre = tre / image_diagonal
    return tre

def transform_landmarks(landmarks, displacement_field):
    u_x = displacement_field[0, :, :, 1].detach().cpu().numpy()
    u_y = displacement_field[0, :, :, 0].detach().cpu().numpy()
    landmarks_x = landmarks[:, 0]
    landmarks_y = landmarks[:, 1]
    ux = nd.map_coordinates(u_x, [landmarks_y, landmarks_x], mode='nearest')
    uy = nd.map_coordinates(u_y, [landmarks_y, landmarks_x], mode='nearest')
    new_landmarks = np.stack((landmarks_x + ux, landmarks_y + uy), axis=1)
    return new_landmarks