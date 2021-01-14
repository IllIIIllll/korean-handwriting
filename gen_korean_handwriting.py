# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import argparse
from datetime import datetime
import os

from gan.gan import generator

import tensorflow as tf
from PIL import Image

def main():
    now = datetime.now()
    today = now.strftime('%Y-%m-%d__%H-%M-%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/model.h5')
    parser.add_argument('--output', default=f'data/results/{today}/')
    parser.add_argument('--count', default=5)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    noise = tf.random.normal([args.count, 100])

    gen = generator()
    images = gen.predict(noise)
    images = 0.5 * images + 0.5

    for i in range(len(images)):
        image = images[i].reshape(64, 64)
        image = Image.fromarray(image, 'L')
        image.save(os.path.join(args.output, f'result{i+1}.png'))

if __name__ == '__main__':
    main()