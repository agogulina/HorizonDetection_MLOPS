import tensorflow as tf

class MaxMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=2, name="max_mean_io_u", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # transforms one-hot/probability to indexes
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true_idx, y_pred_idx, sample_weight)