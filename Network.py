#!/usr/bin/env python
# title           :Network.py
# description     :Architecture file(Generator and Discriminator)
# author          :Deepak Birla
# date            :2018/10/30
# usage           :from Network import Generator, Discriminator
# python_version  :3.5.4

# Modules
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add


# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    # model = BatchNormalization(momentum = 0.5)(model)

    model = add([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):

        gen_input1 = Input(shape=self.noise_shape)
        gen_input2 = Input(shape=(None, None, 3))

        conv2d_1 = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")
        p_re_lu_1 = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        # gen_model1 = model1
        #  ResNet invariants definition

        res_1_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_1_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_1_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_2_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_2_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_2_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_3_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_3_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_3_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_4_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_4_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_4_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_5_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_5_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_5_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_6_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_6_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_6_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_7_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_7_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_7_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_8_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_8_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_8_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_9_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_9_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_9_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_10_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_10_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_10_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_11_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_11_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_11_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_12_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_12_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_12_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_13_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_13_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_13_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_14_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_14_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_14_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_14_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_14_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_14_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_15_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_15_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_15_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        res_16_conv2d_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        res_16_p_re_lu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        res_16_conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        # finish

        conv2d_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")

        up_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        up_2 = UpSampling2D(size=2)
        up_3 = LeakyReLU(alpha=0.2)

        conv2d_3 = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")

        # model = add([gen, model])
        # # Using 16 Residual Blocks
        # for index in range(16):
        #     model1 = res_block_gen(model1, 3, 64, 1)
        #
        # model1 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model1)
        # # model = BatchNormalization(momentum = 0.5)(model)
        # model1 = add([gen_model1, model1])
        #
        # # Using 2 UpSampling Blocks
        # for index in range(1):  # Ishikawa's change
        #     model1 = up_sampling_block(model1, 3, 256, 1)
        #
        # model1 = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model1)
        # model1 = Activation('tanh')(model1)
        #
        # generator_model1 = Model(inputs=gen_input1, outputs=model1)

        model1 = conv2d_1(gen_input1)
        model1 = p_re_lu_1(model1)
        gen_model1 = model1

        gen_t = model1
        model1 = res_1_conv2d_1(model1)
        model1 = res_1_p_re_lu(model1)
        model1 = res_1_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_2_conv2d_1(model1)
        model1 = res_2_p_re_lu(model1)
        model1 = res_2_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_3_conv2d_1(model1)
        model1 = res_3_p_re_lu(model1)
        model1 = res_3_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_4_conv2d_1(model1)
        model1 = res_4_p_re_lu(model1)
        model1 = res_4_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_5_conv2d_1(model1)
        model1 = res_5_p_re_lu(model1)
        model1 = res_5_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_6_conv2d_1(model1)
        model1 = res_6_p_re_lu(model1)
        model1 = res_6_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_7_conv2d_1(model1)
        model1 = res_7_p_re_lu(model1)
        model1 = res_7_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_8_conv2d_1(model1)
        model1 = res_8_p_re_lu(model1)
        model1 = res_8_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_9_conv2d_1(model1)
        model1 = res_9_p_re_lu(model1)
        model1 = res_9_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_10_conv2d_1(model1)
        model1 = res_10_p_re_lu(model1)
        model1 = res_10_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_11_conv2d_1(model1)
        model1 = res_11_p_re_lu(model1)
        model1 = res_11_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_12_conv2d_1(model1)
        model1 = res_12_p_re_lu(model1)
        model1 = res_12_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_13_conv2d_1(model1)
        model1 = res_13_p_re_lu(model1)
        model1 = res_13_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_14_conv2d_1(model1)
        model1 = res_14_p_re_lu(model1)
        model1 = res_14_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_15_conv2d_1(model1)
        model1 = res_15_p_re_lu(model1)
        model1 = res_15_conv2d_2(model1)
        model1 = add([gen_t, model1])

        gen_t = model1
        model1 = res_16_conv2d_1(model1)
        model1 = res_16_p_re_lu(model1)
        model1 = res_16_conv2d_2(model1)
        model1 = add([gen_t, model1])

        # # Using 16 Residual Blocks
        # for index in range(16):
        #     model1 = res_block_gen(model1, 3, 64, 1)

        model1 = conv2d_2(model1)
        model1 = add([gen_model1, model1])

        # Using 2 UpSampling Blocks
        # for index in range(1):  # Ishikawa's change
        #     model1 = up_sampling_block(model1, 3, 256, 1)
        model1 = up_1(model1)
        model1 = up_2(model1)
        model1 = up_3(model1)

        model1 = conv2d_3(model1)
        model1 = Activation('tanh')(model1)

        generator_model1 = Model(inputs=gen_input1, outputs=model1)
        generator_model1.summary()

        model2 = conv2d_1(gen_input2)
        model2 = p_re_lu_1(model2)
        gen_model2 = model2

        gen_t = model2
        model2 = res_1_conv2d_1(model2)
        model2 = res_1_p_re_lu(model2)
        model2 = res_1_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_2_conv2d_1(model2)
        model2 = res_2_p_re_lu(model2)
        model2 = res_2_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_3_conv2d_1(model2)
        model2 = res_3_p_re_lu(model2)
        model2 = res_3_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_4_conv2d_1(model2)
        model2 = res_4_p_re_lu(model2)
        model2 = res_4_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_5_conv2d_1(model2)
        model2 = res_5_p_re_lu(model2)
        model2 = res_5_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_6_conv2d_1(model2)
        model2 = res_6_p_re_lu(model2)
        model2 = res_6_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_7_conv2d_1(model2)
        model2 = res_7_p_re_lu(model2)
        model2 = res_7_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_8_conv2d_1(model2)
        model2 = res_8_p_re_lu(model2)
        model2 = res_8_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_9_conv2d_1(model2)
        model2 = res_9_p_re_lu(model2)
        model2 = res_9_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_10_conv2d_1(model2)
        model2 = res_10_p_re_lu(model2)
        model2 = res_10_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_11_conv2d_1(model2)
        model2 = res_11_p_re_lu(model2)
        model2 = res_11_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_12_conv2d_1(model2)
        model2 = res_12_p_re_lu(model2)
        model2 = res_12_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_13_conv2d_1(model2)
        model2 = res_13_p_re_lu(model2)
        model2 = res_13_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_14_conv2d_1(model2)
        model2 = res_14_p_re_lu(model2)
        model2 = res_14_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_15_conv2d_1(model2)
        model2 = res_15_p_re_lu(model2)
        model2 = res_15_conv2d_2(model2)
        model2 = add([gen_t, model2])

        gen_t = model2
        model2 = res_16_conv2d_1(model2)
        model2 = res_16_p_re_lu(model2)
        model2 = res_16_conv2d_2(model2)
        model2 = add([gen_t, model2])

        # # Using 16 Residual Blocks
        # for index in range(16):
        #     model1 = res_block_gen(model1, 3, 64, 1)

        model2 = conv2d_2(model2)
        model2 = add([gen_model2, model2])

        # Using 2 UpSampling Blocks
        # for index in range(1):  # Ishikawa's change
        #     model1 = up_sampling_block(model1, 3, 256, 1)
        model2 = up_1(model2)
        model2 = up_2(model2)
        model2 = up_3(model2)

        model2 = conv2d_3(model2)
        model2 = Activation('tanh')(model2)

        generator_model2 = Model(inputs=gen_input2, outputs=model2)

        return generator_model1, generator_model2


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Discriminator(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        dis_input = Input(shape=self.image_shape)

        model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)

        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model


def test_Generator():
    gen_input = Input(shape=(None, None, 3))

    model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        model)
    # model = LeakyReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
    # model)
    gen_model = model

    # Using 16 Residual Blocks
    for index in range(16):
        model = res_block_gen(model, 3, 64, 1)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    # model = BatchNormalization(momentum = 0.5)(model)
    model = add([gen_model, model])

    # Using 2 UpSampling Blocks
    for index in range(1):  # Ishikawa's change
        model = up_sampling_block(model, 3, 256, 1)

    model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
    model = Activation('tanh')(model)

    generator_model = Model(inputs=gen_input, outputs=model)
    return generator_model
