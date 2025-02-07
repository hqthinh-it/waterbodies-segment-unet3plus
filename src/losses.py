"""
Implementation of different loss functions
"""
import tensorflow as tf

num_classes = 2

def init_num_classes(init_num_classes):
    global num_classes
    num_classes = init_num_classes

def iou(y_true, y_pred, smooth=tf.keras.backend.epsilon()):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes=num_classes)[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true_f * y_pred_f), axis=-1)
    union = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1) - intersection
    iou = tf.keras.backend.mean((intersection + smooth) / (union + smooth))
    return iou

def dice_coef(y_true, y_pred, smooth=tf.keras.backend.epsilon()):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes=num_classes)[..., 1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1:])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1)
    return tf.keras.backend.mean((2. * intersect) / (denom + smooth))

class dice_coef_loss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        dice = 1 - dice_coef(y_true, y_pred)
        return dice
