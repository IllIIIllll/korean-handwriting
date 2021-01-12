# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from tensorflow.keras.optimizers import Adam

def generate_optimizer(learning_rate):
    return Adam(learning_rate), Adam(learning_rate)
