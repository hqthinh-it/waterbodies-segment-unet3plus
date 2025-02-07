"""
UNet base model
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(input_tensor, num_filters):
    """ Convolutional Block """
    x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(input_tensor)
    x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def unet(input_shape, num_classes):
    """ UNet Model """
    filters = [64, 128, 256, 512, 1024]
    input_layer = layers.Input(shape=input_shape, name="input_layer")

    # Encoder
    e1 = conv_block(input_layer, filters[0])
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = conv_block(p1, filters[1])
    p2 = layers.MaxPooling2D((2, 2))(e2)

    e3 = conv_block(p2, filters[2])
    p3 = layers.MaxPooling2D((2, 2))(e3)

    e4 = conv_block(p3, filters[3])
    p4 = layers.MaxPooling2D((2, 2))(e4)

    # Bottleneck
    b = conv_block(p4, filters[4])

    # Decoder
    d4 = layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding="same")(b)
    d4 = layers.concatenate([d4, e4])
    d4 = conv_block(d4, filters[3])

    d3 = layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding="same")(d4)
    d3 = layers.concatenate([d3, e3])
    d3 = conv_block(d3, filters[2])

    d2 = layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding="same")(d3)
    d2 = layers.concatenate([d2, e2])
    d2 = conv_block(d2, filters[1])

    d1 = layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding="same")(d2)
    d1 = layers.concatenate([d1, e1])
    d1 = conv_block(d1, filters[0])

    # Output Layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d1)

    return Model(inputs=input_layer, outputs=outputs, name="UNet")

