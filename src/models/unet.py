import tensorflow as tf
from src.evaluation.losses import dice_loss
from src.evaluation.metrics import MaxMeanIoU

def create_unet_model(image_size: tuple, 
                      num_classes: int, 
                      learning_rate: float,
                      n_encoder_decoder: int, 
                      initial_filters: int) -> tf.keras.Model:
    """
    Creates a U-Net model for segmentation.
    All hyperparameters are passed from the cfg.
    """
    inputs = tf.keras.layers.Input(shape=(*image_size, 3))
    x = inputs
    encoder_outputs = []

    # Encoder
    for i in range(n_encoder_decoder):
        filters = int(initial_filters * (2 ** i))
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        encoder_outputs.append(x)
        x = tf.keras.layers.MaxPool2D()(x)

    # Bridge
    bridge_filters = int(initial_filters * (2 ** n_encoder_decoder))
    x = tf.keras.layers.Conv2D(bridge_filters, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(bridge_filters, 3, activation='relu', padding='same')(x)

    # Decoder
    for i in reversed(range(n_encoder_decoder)):
        filters = int(initial_filters * (2 ** i))
        x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, activation='relu', padding='same')(x)
        x = tf.keras.layers.Concatenate(axis=3)([x, encoder_outputs[i]])
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    
    # Output head
    outputs = tf.keras.layers.Conv2D(num_classes, 1)(x)
    outputs = tf.keras.layers.Activation('softmax')(outputs)

    # Build & Compile
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=dice_loss,
        metrics=["accuracy", MaxMeanIoU(num_classes=num_classes)]
    )
    
    return model