import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
from sys import platform
from string import ascii_letters
import random
from PIL import Image, ImageFont, ImageDraw, ImageFilter

if platform == 'linux':
    serif = ['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc']
else:
    serif = ['simhei.ttf', 'simhei.ttf']


def load_datasets(root_dir, redux, batch_size, param, crop_size, shuffled=False, single=False):
    dataset = NoiseDataset(root_dir, redux, param, crop_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffled)

class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)

class NoiseDataset(AbstractDataset):
    def __init__(self, root_dir, redux, noise_param, crop_size, seed=None):
        """Initializes noisy image dataset."""

        super(NoiseDataset, self).__init__(root_dir, redux, crop_size)

        self.imgs = os.listdir(root_dir)
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)
        self.noise_param = noise_param

    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)

        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            is_ascii = random.randint(0, 1)
            if is_ascii:
                font = ImageFont.truetype(serif[0], np.random.randint(8, 60))
                length = np.random.randint(10, 25)
                chars = ''.join(random.choice(ascii_letters) for i in range(length))
            else:
                font = ImageFont.truetype(serif[1], np.random.randint(8, 90))
                length = np.random.randint(10, 25)
                chars = ''.join(chr(random.randint(0x4E00, 0x9FBF)) for i in range(length))
            is_color = random.randint(0,2)
            if is_color == 0:
                color = tuple(np.random.randint(0, 1, c))
            elif is_color == 2:
                color = tuple(np.random.randint(255, 256, c))
            else:
                color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img

    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        return self._add_text_overlay(img)

    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img = Image.open(img_path).convert('RGB')

        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        tmp = self._corrupt(img)
        tmp = tmp.filter(ImageFilter.SMOOTH_MORE)
        source = tvF.to_tensor(tmp)
        source = T.Normalize(mean = (0.5,0.5,0.5), std = (1,1,1))(source)
        # Corrupt target image, but not when clean targets are requested
        target = tvF.to_tensor(img.filter(ImageFilter.SMOOTH_MORE))
        target = T.Normalize(mean = (0.5,0.5,0.5), std = (1,1,1))(target)
        return source, target
