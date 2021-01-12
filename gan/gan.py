# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape,
                                     Conv2DTranspose)

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
