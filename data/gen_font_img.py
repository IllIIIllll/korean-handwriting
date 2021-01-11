# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import glob
import os
from tqdm import tqdm

from PIL import ImageFont, Image, ImageDraw
from tensorflow.keras.preprocessing import image
import numpy as np

from utils.params import *

def dec_to_hex(n):
    return hex(n).replace('0x', '').upper()

def hex_to_kr(h):
    return chr(int(h, 16))

def get_font_name(font):
    return os.path.basename(font).split('.')[0]

def font_to_img_array(uni, font):
    img = Image.new('L', (64, 64), color=255)
    draw = ImageDraw.Draw(img)
    kr = hex_to_kr(uni)
    draw.text((32, 32), kr, fill=0, anchor='mm', font=font)
    return image.img_to_array(img, data_format='channels_last')

def main():
    fonts = glob.glob('fonts/*.ttf')
    kr_unicords = [dec_to_hex(i) for i in range(44032, 55204)]

    if not os.path.exists(IMG_PATH):
        os.makedirs(IMG_PATH)

    font_imgs = []
    for font in tqdm(fonts):
        font_name = get_font_name(font).replace(' ', '_')
        font = ImageFont.truetype(font, 50)

        for unicode in kr_unicords:
            img_array = font_to_img_array(unicode, font)
            font_imgs.append(img_array)

        np.save(
            os.path.join(IMG_PATH, font_name + '.npy'),
            np.array(font_imgs)
        )
        font_imgs = []

if __name__ == '__main__':
    main()