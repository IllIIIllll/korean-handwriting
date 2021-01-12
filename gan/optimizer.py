# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like

def generate_optimizer(learning_rate):
    return Adam(learning_rate), Adam(learning_rate)

def generator_loss(fake):
    cross_entropy = BinaryCrossentropy(from_logits=True)
    return cross_entropy(ones_like(fake), fake)
