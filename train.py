# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

from gan.optimizer import *

from tensorflow import GradientTape

def train_step(gen, dsc, gen_opt, disc_opt, font_images, noise):
    with GradientTape() as gen_tape, GradientTape() as dsc_tape:
        generated_image = gen(noise, training=True)

        real = dsc(font_images, training=True)
        fake = dsc(generated_image, training=True)

        gen_loss = generator_loss(fake)
        dsc_loss = discriminator_loss(real, fake)

    gen_gradient = gen_tape.gradient(gen_loss, gen.trainable_variables)
    disc_gradient = dsc_tape.gradient(dsc_loss, dsc.trainable_variables)

    gen_opt.apply_gradients(zip(gen_gradient, gen.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradient, dsc.trainable_variables))
