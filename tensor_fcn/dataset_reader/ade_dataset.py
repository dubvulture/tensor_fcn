from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np

from tensor_fcn.dataset_reader.dataset_reader import BatchDataset


class ADE_Dataset(BatchDataset):

    def __init__(self,
                 ade_dir,
                 batch_size,
                 image_size,
                 fnames=None,
                 crop=True,
                 augment_data=False):
        """
        :param ade_dir: ADEChallengeData2016 directory
        :param batch_size:
        :param image_size: crop window size
        :param fnames: list of filenames (if only a subset has to be selected)
        :param crop: whether to crop images to image_size
        :param augment_data: whether to further augment data

        Here's some examles
            - crop = True, image_size = X, batch_size = Y
                Load images and crop them to image_size
            - crop = False, image_size = 0, batch_size = 1
                Load images from storage and do not crop them
        """
        fnames = fnames or [i[:-4] for i in os.listdir(os.path.join(ade_dir, 'images/'))]

        crop_fun = self._crop_resize if crop else None
        BatchDataset.__init__(self,
                              fnames,
                              batch_size,
                              image_size,
                              image_op=crop_fun,
                              augment_data=augment_data)
        self.ade_dir = ade_dir

    def _get_image(self, fname):
        """
        Load image already saved on the disk
        """
        image = cv2.imread(
                os.path.join(self.ade_dir, 'images/', fname + '.jpg'))
        annotation = cv2.imread(
                os.path.join(self.ade_dir, 'annotations/', fname + '.png'))
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
        weight = np.ones(annotation.shape, dtype=np.float32)

        return [image, annotation, weight]

    def _crop_resize(self, image, annotation, weight, name=None):
        x1 = x2 = np.random.randint(0, image.shape[1]-1)
        x2 += 1
        y1 = y2 = np.random.randint(0, image.shape[0]-1)
        y2 += 1
        h, w = image.shape[:2]

        # expand left, right, up, down
        x1 -= np.random.randint(0, x1 + 1)
        x2 += np.random.randint(0, w - x2 + 1)
        y1 -= np.random.randint(0, y1 + 1)
        y2 += np.random.randint(0, h - y2 + 1)

        ratio = (x2 - x1) / (y2 - y1)

        if ratio > 1:
            # expand ys (window's height)
            diff = (x2 - x1) - (y2 - y1)
            split = np.random.randint(0, diff)
            y1 -= split
            y2 += diff - split
        elif ratio < 1:
            # expand xs (window's width)
            diff = (y2 - y1) - (x2 - x1)
            split = np.random.randint(0, diff)
            x1 -= split
            x2 += diff - split
        else:
            pass

        assert((x2 - x1) == (y2 - y1)), '%d, %d, %d, %d' % (x1,x2,y1,y2)

        slices = [
                slice(max(0, y1), y2),
                slice(max(0, x1), x2)
        ]
        pads = (
                (max(0, -y1), max(0, y2-h)),
                (max(0, -x1), max(0, x2-w))
        )

        images = [image, annotation, weight]
        pad = [((0, 0),), (), ()]
        interp = [cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_NEAREST]
        dsize = (self.image_size, self.image_size)
        value = [0.0, 0.0, 1.0]

        for i in range(3):
            images[i] = images[i][slices]
            images[i] = np.pad(images[i],
                               pads + pad[i],
                               'constant',
                               constant_values=value[i])
            images[i] = cv2.resize(images[i], dsize, interpolation=interp[i])

        return images
