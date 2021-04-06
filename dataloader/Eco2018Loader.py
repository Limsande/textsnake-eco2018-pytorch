import numpy as np
import os
import torch.utils.data as data

from utils import to_device, get_device, load_img_as_np_array


class RootDataset(data.Dataset):
    """
    Only  implements some basic preparations to put images into the neural net.
    Any subclass has to take care of loading images and calculating all the input
    for TextSnake, like center line, root polygons etc. as well as applying
    augmentation.
    """

    def __getitem__(self, index):
        raise NotImplementedError()

    def __init__(self):
        super().__init__()

    def get_training_data(self, img_and_masks):
        """
        Prepares meta data the network needs for training.

        :param img_and_masks: dictionary with input image and one mask per TextSnake input
        """

        img_height, img_width, _ = img_and_masks['img'].shape

        # to pytorch channel sequence
        img_and_masks['img'] = img_and_masks['img'].transpose(2, 0, 1)

        # All input images are uint8. Do some type conversions
        # to match expected model input:
        #   Image: float32
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

        img_and_masks['img'] = img_and_masks['img'].astype(np.float32)
        img_and_masks['radii'] = img_and_masks['radii'].astype(np.float32)

        # Map [0, 255] int to [-1, 1] float
        for key in ['sin', 'cos']:
            map = img_and_masks[key].astype(np.float32)
            map -= 255 / 2  # [0, 255] -> [-127.5, 127.5]
            map /= 255 / 2  # [-127.5, 127.5] -> [-1, 1]
            img_and_masks[key] = map

        return (img_and_masks['img'],
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
        image = load_img_as_np_array(image_path)

        # Read annotations and build a dict with them
        img_and_masks = {'img': image}
        for annotation_name in self._annotation_names:
            annotation_id = self.annotation_lists[annotation_name][item]
            annotation_path = os.path.join(self.annotation_root,
                                           annotation_name, annotation_id)
            img_and_masks[annotation_name] = load_img_as_np_array(annotation_path)

        # Apply augmentations to image and masks
        if self.transformations:
            img_and_masks = self.transformations(img_and_masks)

        # image, tr_mask, tcl_mask, radius_map, sin_map, cos_map
        return self.get_training_data(img_and_masks)

    def __len__(self):
        return len(self.image_list)


class DeviceLoader:
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
