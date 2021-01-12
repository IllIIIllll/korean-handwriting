# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape,
                                     Conv2DTranspose, Conv2D, Dropout, Flatten)

def generator():
    gen = Sequential([
        Dense(8 * 8 * 512, use_bias=False, input_dim=100),
        BatchNormalization(),
        LeakyReLU(),
        Reshape((8, 8, 512)),

        Conv2DTranspose(
            128, (5, 5),
            strides=(1, 1),
            padding='same',
            use_bias=False
        ),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(
            64, (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False
        ),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(
            32, (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False
        ),
        BatchNormalization(),
        LeakyReLU(),

        Conv2DTranspose(
            1, (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            activation='tanh'
        )
    ])
    return gen

def discriminator():
    dsc = Sequential([
        Conv2D(32, (5, 5), strides=2, padding='same', input_shape=[64, 64, 1]),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(64, (5, 5), strides=2, padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Conv2D(128, (5, 5), strides=2, padding='same'),
        LeakyReLU(),
        Dropout(0.3),

        Flatten(),
        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.3),

        Dense(1)
    ])
    return dsc