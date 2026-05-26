

import random
import numbers
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import Compose, Lambda


# =========================
# basic utils
# =========================
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    img: np.float32 (3,T,H,W) in [0,255]
    return: np.float32 (1,T,H,W) in [0,255]
    """
    if not isinstance(img, np.ndarray) or img.ndim != 4 or img.shape[0] != 3:
        raise ValueError(f"rgb_to_gray expects (3,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

    _, T, H, W = img.shape
    out = np.empty((1, T, H, W), dtype=np.float32)

    for n in range(T):
        frame = np.transpose(img[:, n], (1, 2, 0))  # HWC
        pil = Image.fromarray(np.uint8(np.clip(frame, 0, 255)))
        pil = pil.convert('L')
        out[0, n] = np.asarray(pil, dtype=np.float32)

    return out


def histeq(img: np.ndarray) -> np.ndarray:
    """
    histogram equalization for grayscale video
    img: np.float32 (1,T,H,W) in [0,255]
    """
    if not isinstance(img, np.ndarray) or img.ndim != 4 or img.shape[0] != 1:
        raise ValueError(f"histeq expects (1,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

    _, T, H, W = img.shape
    out = np.empty_like(img, dtype=np.float32)

    for n in range(T):
        frame = img[0, n]
        imhist, bins = np.histogram(frame.flatten(), 256, range=(0, 255))
        cdf = imhist.cumsum()
        cdf = 255.0 * cdf / (cdf[-1] + 1e-8)
        im = np.interp(frame.flatten(), bins[:-1], cdf).reshape(frame.shape)
        out[0, n] = im.astype(np.float32)

    return out


def normalize(image: torch.Tensor,
              mean=(0.43216, 0.394666, 0.37645),
              std=(0.22803, 0.22145, 0.216989)) -> torch.Tensor:
    """
    image: torch float tensor (C,T,H,W) or (C,H,W), usually in [0,1]
    """
    if not torch.is_tensor(image):
        raise TypeError(f"normalize expects torch.Tensor, got {type(image)}")
    if image.dim() not in (3, 4):
        raise ValueError(f"normalize expects 3D/4D tensor, got {tuple(image.shape)}")

    C = image.shape[0]
    if C != len(mean) or C != len(std):
        raise ValueError(f"channel mismatch: C={C}, mean={len(mean)}, std={len(std)}")

    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(C, *([1] * (image.dim() - 1)))
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(C, *([1] * (image.dim() - 1)))
    return (image - mean_t) / std_t


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    img: np.float32 (C,T,H,W) in [0,255]
    return: torch.float32 (C,T,H,W) in [0,1]
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"to_tensor expects np.ndarray, got {type(img)}")
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    return torch.from_numpy(img).float() / 255.0


class ToTensor(object):
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return to_tensor(img)


def _chw_to_pil(frame_chw: np.ndarray) -> Image.Image:
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
    frame_u8 = np.uint8(np.clip(frame_hwc, 0, 255))
    return Image.fromarray(frame_u8)


def _pil_to_chw(pil: Image.Image) -> np.ndarray:
    arr = np.asarray(pil, dtype=np.float32)     # HWC
    return np.transpose(arr, (2, 0, 1))         # CHW


# =========================
# Augmentations (numpy in/out)
# =========================
class RandomHorizontalFlip(object):
    """Sequence-consistent flip on numpy (C,T,H,W)."""
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return img

        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"RandomHorizontalFlip expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        C, T, H, W = img.shape
        out = img.copy()
        for n in range(T):
            pil = _chw_to_pil(out[:, n])
            pil = F.hflip(pil)
            out[:, n] = _pil_to_chw(pil)
        return out


class CenterCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"CenterCrop expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        C, T, H, W = img.shape
        out = img.copy()
        for n in range(T):
            pil = _chw_to_pil(out[:, n])
            pil = F.center_crop(pil, self.size)
            out[:, n] = _pil_to_chw(pil)
        return out


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"Resize expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        C, T, H, W = img.shape
        out = img.copy()
        for n in range(T):
            pil = _chw_to_pil(out[:, n])
            pil = F.resize(pil, self.size, self.interpolation)
            out[:, n] = _pil_to_chw(pil)
        return out


class Noise(object):
    """Add Gaussian noise on numpy (C,T,H,W)."""
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"Noise expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        gauss = np.random.normal(0, self.sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + gauss
        out = np.clip(out, 0, 255).astype(np.float32)
        return out


class Blurring(object):
    """Gaussian blur (per frame)."""
    def __init__(self, size):
        k = int(size)
        if k <= 0 or k % 2 == 0:
            raise ValueError("Blurring size must be positive odd integer (e.g., 3/5/7)")
        self.size = k

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"Blurring expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        C, T, H, W = img.shape
        out = img.copy().astype(np.float32)
        for n in range(T):
            frame = np.transpose(out[:, n], (1, 2, 0))  # HWC
            frame = cv2.GaussianBlur(frame, (self.size, self.size), 0)
            out[:, n] = np.transpose(frame.astype(np.float32), (2, 0, 1))
        out = np.clip(out, 0, 255).astype(np.float32)
        return out


class ColorJitter(object):
    """Sequence-consistent color jitter on numpy (C,T,H,W)."""
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 4:
            raise ValueError(f"ColorJitter expects (C,T,H,W) numpy, got {type(img)} {getattr(img, 'shape', None)}")

        transforms = []

        # sample ONCE per clip (sequence-consistent)
        if self.brightness is not None:
            bf = random.uniform(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda im: F.adjust_brightness(im, bf)))
        if self.contrast is not None:
            cf = random.uniform(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda im: F.adjust_contrast(im, cf)))
        if self.saturation is not None:
            sf = random.uniform(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda im: F.adjust_saturation(im, sf)))
        if self.hue is not None:
            hf = random.uniform(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda im: F.adjust_hue(im, hf)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        C, T, H, W = img.shape
        out = img.copy().astype(np.float32)
        for n in range(T):
            pil = _chw_to_pil(out[:, n])
            pil = transform(pil)
            out[:, n] = _pil_to_chw(pil)
        return out
