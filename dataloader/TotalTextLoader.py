"""
From https://github.com/princewang1994/TextSnake.pytorch
"""
import copy

import cv2
import math
import numpy as np
import os
import scipy.io as io
from skimage.draw import polygon as drawpoly
from torch.utils import data

from utils import load_img_as_np_array


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))


def vector_sin(v):
    assert len(v) == 2
    # sin = y / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[1] / l


def vector_cos(v):
    assert len(v) == 2
    # cos = x / (sqrt(x^2 + y^2))
    l = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return v[0] / l


def find_bottom(pts):
    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i + 2] - e[i + 1]
            if cos(v_prev, v_next) < -0.7:
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]

    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms


def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while (i % n_pts != b2_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while (i % n_pts != b1_end):
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_edge_seqence(points, long_edge, n_parts):
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while (cur_end > point_cumsum[cur_node + 1]):
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        self.transform = transform

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4:  # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        # TODO size
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=(512, 512))
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, radius, \
                              tcl_mask, radius_map, sin_map, cos_map, expand=0.3, shrink=1):

        for i in range(shrink, len(center_line) - 1 - shrink):
            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            sin_theta = vector_sin(c2 - c1)
            cos_theta = vector_cos(c2 - c1)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand
            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)
            self.fill_polygon(radius_map, polygon, value=radius[i])
            self.fill_polygon(sin_map, polygon, value=sin_theta)
            self.fill_polygon(cos_map, polygon, value=cos_theta)

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        tcl_mask = np.zeros(image.shape[:2], np.uint8)
        radius_map = np.zeros(image.shape[:2], np.float32)
        sin_map = np.zeros(image.shape[:2], np.float32)
        cos_map = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(n_disk=15)
                self.make_text_center_line(sideline1, sideline2, center_points, radius, tcl_mask, radius_map, sin_map,
                                           cos_map)
        tr_mask, train_mask = self.make_text_region(image, polygons)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        points = np.zeros((200, 20, 2))
        length = np.zeros(200, dtype=int)

        for i, polygon in enumerate(polygons):
            pts = polygon.points
            points[i, :pts.shape[0]] = polygon.points
            length[i] = pts.shape[0]

        return image, tr_mask, tcl_mask, radius_map, sin_map, cos_map

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

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


class TotalText(TextDataset):

    def __init__(self, data_root='data/Total-Text', is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '')) for img_name in self.image_list]

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = load_img_as_np_array(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            Padding(),
            RandomResizedLimitCrop(size=size, scale=(0.24, 1.0), ratio=(0.33, 3)),
            # RandomBrightness(),
            # RandomContrast(),
            RandomMirror(),
            Rotate(),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            return image, polygons

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 2)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = polygon.points[:, 0] + left
                polygon.points[:, 1] = polygon.points[:, 1] + top
        return image, polygons


class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, polygons=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        cropped = image[i:i + h, j:j + w, :]
        scales = np.array([self.size[0] / w, self.size[1] / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points[:, 0] = (polygon.points[:, 0] - j) * scales[0]
                polygon.points[:, 1] = (polygon.points[:, 1] - i) * scales[1]

        img = cv2.resize(cropped, self.size)
        return img, polygons


class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons


class Rotate(object):
    def __init__(self, up=30):
        self.up = up

    def rotate(self, center, pt, theta):  # 二维图形学的旋转 Rotation of 2D graphics
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

    def __call__(self, img, polygons=None):
        if np.random.randint(2):
            return img, polygons
        angle = np.random.uniform(-self.up, self.up)  #
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
        center = cols / 2.0, rows / 2.0
        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle)
                pts = np.vstack([x, y]).T
                polygon.points = pts
        return img, polygons


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


if __name__ == '__main__':

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='data/total-text',
        # ignore_list='./ignore_list.txt',
        is_training=True,
        transform=transform
    )

    # img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[944]

    for idx in range(0, len(trainset)):
        img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, meta = trainset[idx]
        print('Image:', idx, img.shape)
        print('Train mask:', train_mask.shape, train_mask.dtype)
        print('TR mask:', tr_mask.shape, tr_mask.dtype)
        print('TCL mask:', tcl_mask.shape, tcl_mask.dtype)
        print('Radius map:', radius_map.shape, radius_map.dtype)
        print('Sin map:', sin_map.shape, sin_map.dtype)
        print('Cos map:', cos_map.shape, cos_map.dtype)
