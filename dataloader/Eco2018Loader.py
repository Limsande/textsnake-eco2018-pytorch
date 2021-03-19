import os

import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from utils import to_device, get_device


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class RootInstance(object):
    """
    Represents a single root as a polygon.
    """

    def __init__(self, points: np.array):
        """
        :param points: Nx2 numpy array, where N is number of points
            in this root (i.e. polygon)
        """
        remove_points = []

        # Try to reduce number of points in this polygon without
        # loosing to much information.
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(
                        points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if
                                    i not in remove_points])
        else:
            self.points = np.array(points)

        self.length = len(self.points)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


def roots_to_polygons(annotation_mask) -> [RootInstance]:
    """
    Extracts roots as polygons from binary annotation mask.
    """

    # With given options, cv.findContours() only supports uint8 input.
    if annotation_mask.dtype is not np.uint8:
        annotation_mask = annotation_mask * 255
        annotation_mask = annotation_mask.astype(np.uint8)

    # Retrieval mode = cv.RETR_EXTERNAL: find outer contours only,
    # no hierarchy established;
    # Contour approximation method = cv.CHAIN_APPROX_SIMPLE: do not
    # store *every* point on the contour, only important ones
    contours, _ = cv.findContours(annotation_mask, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)

    # list of contours, each is a Nx1x2 numpy array,
    # where N is number of points. Remove intermediate
    # dimension of length 1
    contours = [RootInstance(points=c.squeeze()) for c in contours]

    return contours


class RootDataset(data.Dataset):
    """
    Only  implements some basic preparations to put images into the neural net.
    Any subclass has to take care of loading images and calculating all the input
    for TextSnake, like center line, root polygons etc. as well as applying
    augmentation.
    """

    def __init__(self):
        super().__init__()

    def get_training_data(self, img_and_masks, polygons, image_id, image_path):
        """
        Prepares meta data the network needs for training.

        :param img_and_masks: dictionary with input image and one mask per TextSnake input
        :param polygons: list of RootInstance objects defining the roots in this img_and_masks['img']
        """

        img_height, img_width, _ = img_and_masks['img'].shape

        # train_mask = self.make_text_region(image, polygons)
        # Extracted from make_text_region. No idea what this is for
        train_mask = np.ones(img_and_masks['img'].shape[:2], np.uint8)

        # to pytorch channel sequence
        img_and_masks['img'] = img_and_masks['img'].transpose(2, 0, 1)

        # max_annotation = max #polygons per image
        # max_points = max #points per polygons
        max_points = max([p.length for p in polygons])
        max_annotation = 200
        all_possible_points_for_each_possible_polygon = np.zeros(
            (max_annotation, max_points, 2))
        n_points_per_polygon = np.zeros(max_annotation, dtype=int)
        for i, polygon in enumerate(polygons):
            # polygon.length = #points in this polygon
            all_possible_points_for_each_possible_polygon[i,
            :polygon.length] = polygon.points
            n_points_per_polygon[i] = polygon.length

        # All input images are uint8. Do some type conversions
        # to match expected model input:
        #   Train mask: uint8, 0 or 1
        #   Root mask: uint8, 0 or 1
        #   Center line mask: uint8, 0 or 1
        #   Radius map: float32
        #   Sin map: float32, -1.0 to 1.0
        #   Cos map: float32, -1.0 to 1.0
        for mask in [img_and_masks['roots'], img_and_masks['centerlines']]:
            if mask.max() > 1:
                # Store result of array division in int array
                # without type conversions.
                # See https://github.com/numpy/numpy/issues/17221
                np.divide(mask, 255, out=mask, casting='unsafe')

        # PyTorch expects float tensors as input
        img_and_masks['img'] = img_and_masks['img'].astype(np.float32)

        img_and_masks['radii'] = img_and_masks['radii'].astype(np.float32)

        # Map [0, 255] to [-1, 1]
        for key in ['sin', 'cos']:
            map = img_and_masks[key].astype(np.float32)
            map -= 255 / 2  # [0, 255] -> [-127.5, 127.5]
            map /= 255 / 2  # [-127.5, 127.5] -> [-1, 1]
            img_and_masks[key] = map

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': all_possible_points_for_each_possible_polygon,
            'n_annotation': n_points_per_polygon,
            'Height': img_height,
            'Width': img_width
        }

        # return img_and_masks, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta
        return (img_and_masks['img'],
                #train_mask,
                img_and_masks['roots'],
                img_and_masks['centerlines'],
                img_and_masks['radii'],
                img_and_masks['sin'],
                img_and_masks['cos']
                )

    def get_test_data(self, image, image_id, image_path):
        # TODO

        H, W, _ = image.shape

        if self.transform:
            # TODO mean and stds
            # image, polygons = self.transform(image)
            pass

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()


class Eco2018(RootDataset):
    """
    Iterable to be passed into PyTorch's data.DataLoader.

    This class loads the images and masks, and extracts root polygons from the binary annotation
    mask (we must feed these into the net). Additional stuff like image type conversions etc.
    is handled by the baseclass.

    Input is expected to follow this structure:

        ./data
            \ Eco2018
                \ annotation
                    \ training
                        \ roots
                        \ ...
                    \ validation
                        \ ...
                \ images
                    \ training
                    \ validation

    Eco2018 differs from Total-Text, because the input for TextSnake, like center lines, is already
    present as additional image masks.
    """

    def __init__(
            self,
            data_root='data/Eco2018',
            is_training=True,
            transformations=None):

        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.transformations = transformations

        self._annotation_names = ['roots', 'centerlines', 'radii', 'sin', 'cos']

        self.image_root = os.path.join(data_root, 'images',
                                       'training' if is_training else 'validation')
        self.annotation_root = os.path.join(data_root, 'annotations',
                                            'training' if is_training else 'validation')

        self.image_list = os.listdir(self.image_root)
        # One list per image with names of root mask, center line mask, etc.
        self.annotation_lists = {
            key: [
                img_name.replace('-', f'-{key}-') for img_name in
                self.image_list
            ] for key in self._annotation_names}

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotations and build a dict with them
        img_and_masks = {'img': image}
        for annotation_name in self._annotation_names:
            annotation_id = self.annotation_lists[annotation_name][item]
            annotation_path = os.path.join(self.annotation_root,
                                           annotation_name, annotation_id)
            img_and_masks[annotation_name] = pil_load_img(annotation_path)

        # Apply augmentations to image and masks
        if self.transformations:
            img_and_masks = self.transformations(img_and_masks)

        polygons = roots_to_polygons(img_and_masks['roots'])

        # image, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta
        return self.get_training_data(img_and_masks, polygons,
                                      image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


class DeviceLoader():
    """
    Thin wrapper around a PyTorch DataLoader moving
    data to the GPU (if available).
    """

    def __init__(self, data_loader):
        self._loader = data_loader
        self._device = get_device()

    def __iter__(self):
        """Move the current batch to device and return it."""
        for batch in self._loader:
            yield to_device(batch, self._device)

    def __len__(self):
        return len(self._loader)
