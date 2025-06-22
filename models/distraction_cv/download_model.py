#!/usr/bin/env python3
"""
Script to download or create a pre-trained VGG16 model for driver distraction detection.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np

def create_vgg16_model():
    """
    Create a VGG16 model with custom top layers for 10-class driver distraction detection.
    This creates a model with the same architecture as the original but with random weights.
    """
    print("🏗️ Creating VGG16 model for driver distraction detection...")
    
    # Load VGG16 pretrained on ImageNet
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    return model

def save_model():
    """Save the model to the expected location."""
    # Create directories if they don't exist
    model_dir = "model/self_trained"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create and save the model
    model = create_vgg16_model()
    model_path = os.path.join(model_dir, "distracted-23-1.00.hdf5")
    model.save(model_path)
    
    print(f"✅ Model saved to: {model_path}")
    return model_path

def create_pickle_files():
    """Create the required pickle files for labels."""
    import pickle
    
    # Create pickle directory
    pickle_dir = "pickle_files"
    os.makedirs(pickle_dir, exist_ok=True)
    
    # Create labels mapping
    labels_id = {
        'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4,
        'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9
    }
    
    # Save labels pickle file
    pickle_path = os.path.join(pickle_dir, "labels_list.pkl")
    with open(pickle_path, "wb") as handle:
        pickle.dump(labels_id, handle)
    
    print(f"✅ Labels pickle file saved to: {pickle_path}")

def main():
    """Main function to set up the model and required files."""
    print("🚗 Setting up VGG16 Driver Distraction Detection Model")
    print("=" * 60)
    
    try:
        # Create model and save it
        model_path = save_model()
        
        # Create pickle files
        create_pickle_files()
        
        print("\n🎉 Setup completed successfully!")
        print(f"📁 Model: {model_path}")
        print("📁 Labels: pickle_files/labels_list.pkl")
        print("\n📝 Note: This is a placeholder model with random weights.")
        print("   For production use, you should train the model on the actual dataset.")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 