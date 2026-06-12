import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice Loss"""
    # y_true, y_pred: (batch, h, w, num_classes)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice)