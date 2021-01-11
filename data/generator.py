# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import numpy as np

class DataGenerator:
    def __init__(self, samples):
        self.samples = samples

    def generate(self, batch_size):
        for sample in self.samples:
            font = np.load(sample)

            while font.shape[0] >= batch_size:
                batch, font = font[:batch_size], font[batch_size:]
                yield batch