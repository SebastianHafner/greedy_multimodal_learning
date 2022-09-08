import torchvision.transforms.functional as TF
import numpy as np


class Numpy2Torch(object):
    def __call__(self, args):
        img_sar, img_optical, label = args
        img_sar_tensor = TF.to_tensor(img_sar)
        img_optical_tensor = TF.to_tensor(img_optical)
        label_tensor = TF.to_tensor(label)
        return img_sar_tensor, img_optical_tensor, label_tensor


class RandomFlip(object):
    def __call__(self, args):
        img_sar, img_optical, label = args
        horizontal_flip = np.random.choice([True, False])
        vertical_flip = np.random.choice([True, False])

        if horizontal_flip:
            img_sar = np.flip(img_sar, axis=1)
            img_optical = np.flip(img_optical, axis=1)
            label = np.flip(label, axis=1)

        if vertical_flip:
            img_sar = np.flip(img_sar, axis=0)
            img_optical = np.flip(img_optical, axis=0)
            label = np.flip(label, axis=0)

        img_sar = img_sar.copy()
        img_optical = img_optical.copy()
        label = label.copy()

        return img_sar, img_optical, label


class RandomRotate(object):
    def __call__(self, args):
        img_sar, img_optical, label = args
        k = np.random.randint(1, 4)  # number of 90 degree rotations
        img_sar = np.rot90(img_sar, k, axes=(0, 1)).copy()
        img_optical = np.rot90(img_optical, k, axes=(0, 1)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()
        return img_sar, img_optical, label


class ColorShift(object):
    def __init__(self, min_factor: float = 0.5, max_factor: float = 1.5):
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, args):
        img_sar, img_optical, label = args
        factors = np.random.uniform(self.min_factor, self.max_factor, img_optical.shape[-1])
        img_optical_rescaled = np.clip(img_optical * factors[np.newaxis, np.newaxis, :], 0, 1).astype(np.float32)
        return img_sar, img_optical_rescaled, label


class GammaCorrection(object):
    def __init__(self, gain: float = 1, min_gamma: float = 0.25, max_gamma: float = 2):
        self.gain = gain
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, args):
        img_sar, img_optical, label = args
        gamma = np.random.uniform(self.min_gamma, self.max_gamma, img_optical.shape[-1])
        img_optical_gamma = np.clip(np.power(img_optical, gamma[np.newaxis, np.newaxis, :]), 0, 1).astype(np.float32)
        return img_sar, img_optical_gamma, label


class ImageCrop(object):
    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, args):
        img_sar, img_optical, label = args
        m, n, _ = img_sar.shape
        i = 0 if m == self.crop_size else np.random.randint(0, m - self.crop_size)
        j = 0 if n == self.crop_size else np.random.randint(0, n - self.crop_size)
        img_sar_crop = img_sar[i:i + self.crop_size, j:j + self.crop_size, ]
        img_optical_crop = img_optical[i:i + self.crop_size, j:j + self.crop_size, ]
        label_crop = label[i:i + self.crop_size, j:j + self.crop_size, ]
        return img_sar_crop, img_optical_crop, label_crop
