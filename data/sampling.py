# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import random
import os
import glob

class Sampler:
    def __init__(self, data_dir, seed):
        self.data_dir = data_dir
        random.seed(seed)

    def draw_data(self, num_samples=None):
        fonts = glob.glob(os.path.join(self.data_dir, '*.npy'))
        if num_samples is None:
            return fonts
        else:
            sample_set = set()
            while len(sample_set) < num_samples:
                sample = random.choice(fonts)
                if sample not in sample_set:
                    sample_set.add(sample)
            return list(sample_set)