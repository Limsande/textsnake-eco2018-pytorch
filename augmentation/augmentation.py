import cv2
import math
import numpy as np


class RootCompose(object):
    """
    Composes several augmentations together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_and_masks):
        for t in self.transforms:
            img_and_masks = t(img_and_masks)
        return img_and_masks


class RootAugmentation(object):
    """
    Container for several augmentations applied to a collection
    of an image and its annotation masks when called.
    """

    def __init__(self, mean, std, pad_size=512):
        self.mean = mean
        self.std = std
        self.augmentation = RootCompose([
            RootPadding(pad_size),
            RootRandomMirror(),
            RootRandomRotate(),
            RootNormalize(mean, std)
        ])

    def __call__(self, img_and_masks):
        return self.augmentation(img_and_masks)


class RootBaseTransform(object):
    """
    Container for normalization and padding applied to a collection
    of an image and its annotation masks when called, see RootNormalize.
    """

    def __init__(self, mean, std, pad_size=512):
        self.mean = mean
        self.std = std
        self.augmentation = RootCompose([
            RootPadding(pad_size),
            RootNormalize(mean, std)
        ])

    def __call__(self, img_and_masks):
        return self.augmentation(img_and_masks)


class RootPadding(object):
    """
    Surrounds squared images smaller then pad_size x pad_size with zeros.
    Resulting images have dimensions pad_size x pad_size.
    """

    def __init__(self, pad_size):
        self.pad_size = pad_size

    def __call__(self, img_and_masks):
        if not img_and_masks['img'].shape[0] < self.pad_size:
            return img_and_masks

        for key in img_and_masks.keys():
            img = img_and_masks[key]

            if img.ndim > 2:
                dims = (self.pad_size, self.pad_size, img.shape[2])
            else:
                dims = (self.pad_size, self.pad_size)

            padded_img = np.zeros(dims, dtype=img.dtype)

            top_left_x = (self.pad_size - img.shape[1]) // 2
            top_left_y = (self.pad_size - img.shape[0]) // 2
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    padded_img[top_left_y + y][top_left_x + x] = img[y][x]

            img_and_masks[key] = padded_img

        return img_and_masks


class RootNormalize(object):
    """
    Normalizes image with key 'img' in input dictionary.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, img_and_masks):
        image = img_and_masks['img'].astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        img_and_masks['img'] = image

        return img_and_masks


class RootRandomMirror(object):
    """
    Mirrors input images with probability of 0.5.
    """

    def __init__(self):
        pass

    def __call__(self, img_and_masks):
        if np.random.randint(2):
            for key in img_and_masks.keys():
                img_and_masks[key] = np.ascontiguousarray(img_and_masks[key][:, ::-1])
        return img_and_masks


class RootRandomRotate(object):
    """
    Rotates input images with probability of 0.5.
    """

    def __init__(self, up=30):
        self.up = up

    def rotate(self, center, pt, theta):  # Rotation of 2D graphics
        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 360 * 2 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        return _x, -_y

    def __call__(self, img_and_masks):
        if np.random.randint(2):
            return img_and_masks

        angle = np.random.uniform(-self.up, self.up)  #
        rows, cols = img_and_masks['img'].shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        for key in img_and_masks.keys():
            img_and_masks[key] = cv2.warpAffine(img_and_masks[key], M, (cols, rows), borderValue=[0, 0, 0])
        return img_and_masks
