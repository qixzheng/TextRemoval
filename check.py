import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import numpy as np

import os
from string import ascii_letters
import random
from PIL import Image, ImageFont, ImageDraw

serif = ['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc']
def add_text(source, noise_param):
    img = Image.open(source).convert('RGB')
    c = len(img.getbands())

    # Choose font and get ready to draw
    
    text_img = img.copy()
    text_draw = ImageDraw.Draw(text_img)
    # Text binary mask to compute occupancy efficiently
    w, h = img.size
    mask_img = Image.new('1', (w, h))
    mask_draw = ImageDraw.Draw(mask_img)
    # Random occupancy in range [0, p]
    max_occupancy = noise_param


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