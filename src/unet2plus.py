"""
UNet 2 plus base model
"""

# import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

def conv_block(input_tensor, num_filters):
    """Convolutional Block"""
    x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(input_tensor)
    x = layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def unet2plus(input_shape, num_classes):
    """UNetPlusPlus Model"""
    filters = [64, 128, 256, 512, 1024]
    input_layer = layers.Input(shape=input_shape)

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
    d4_0 = layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding="same")(b)
    d4_0 = layers.concatenate([d4_0, e4])
    d4_0 = conv_block(d4_0, filters[3])

    d3_0 = layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding="same")(d4_0)
    e3_resized = layers.Conv2D(filters[2], (1, 1), padding="same")(e3)  # Resize channels of e3
    d3_0 = layers.concatenate([d3_0, e3_resized])
    d3_0 = conv_block(d3_0, filters[2])

    d2_0 = layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding="same")(d3_0)
    e2_resized = layers.Conv2D(filters[1], (1, 1), padding="same")(e2)  # Resize channels of e2
    d2_0 = layers.concatenate([d2_0, e2_resized])
    d2_0 = conv_block(d2_0, filters[1])

    d1_0 = layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding="same")(d2_0)
    e1_resized = layers.Conv2D(filters[0], (1, 1), padding="same")(e1)  # Resize channels of e1
    d1_0 = layers.concatenate([d1_0, e1_resized])
    d1_0 = conv_block(d1_0, filters[0])

    # Output layer
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(d1_0)

    return Model(inputs=input_layer, outputs=outputs, name="UNetPlusPlus")
