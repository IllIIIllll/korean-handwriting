# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import os

from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing import image

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