import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.models import Model
import numpy as np

def build_model(
    num_classes=10,
    input_shape=(224, 224, 3),
    backbone='efficientnetb0',
    dropout_rate=0.5,
    learning_rate=1e-4,
    label_smoothing=0.1
):
    """
    Build a Keras model for driver distraction detection.
    
    Args:
        num_classes: Number of distraction classes (default: 10)
        input_shape: Input image shape (default: 224x224x3)
        backbone: Pre-trained backbone model (efficientnetb0, mobilenetv2, custom)
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        label_smoothing: Label smoothing factor
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation layer
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])
    
    x = data_augmentation(inputs)
    
    # Preprocessing layer
    if backbone.startswith('efficientnet'):
        x = applications.efficientnet.preprocess_input(x)
    elif backbone.startswith('mobilenet'):
        x = applications.mobilenet_v2.preprocess_input(x)
    else:
        # Custom preprocessing for RGB values
        x = x / 255.0
    
    # Backbone model
    if backbone == 'efficientnetb0':
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        x = base_model.output
    elif backbone == 'mobilenetv2':
        base_model = applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
        x = base_model.output
    else:
        # Custom CNN backbone
        x = build_custom_backbone(x)
    
    # Freeze the base model initially (only for pre-trained models)
    if backbone in ['efficientnetb0', 'mobilenetv2']:
        base_model.trainable = False
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def build_custom_backbone(inputs):
    """
    Build a custom CNN backbone for cases where pre-trained models aren't available.
    
    Args:
        inputs: Input tensor
    
    Returns:
        Feature tensor (not a model)
    """
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    return x

def unfreeze_model(model, unfreeze_layers=30):
    """
    Unfreeze the top layers of the base model for fine-tuning.
    
    Args:
        model: Compiled Keras model
        unfreeze_layers: Number of layers to unfreeze from the top
    
    Returns:
        Model with unfrozen layers
    """
    # Get the base model (first layer after input)
    base_model = model.layers[1]
    
    # Unfreeze the top layers
    base_model.trainable = True
    
    # Freeze all the layers before the `unfreeze_layers` layer
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def get_model_summary(model):
    """
    Get a formatted model summary.
    
    Args:
        model: Keras model
    
    Returns:
        String representation of model summary
    """
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    return '\n'.join(summary)

if __name__ == "__main__":
    # Test model creation
    model = build_model()
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print("\nModel Summary:")
    print(get_model_summary(model)) 