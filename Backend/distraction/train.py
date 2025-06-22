#!/usr/bin/env python3
"""
Driver Distraction Detection Training Script
Based on Toshi-K Kaggle Distracted Driver Detection
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import matplotlib.pyplot as plt
from datetime import datetime
import json

from model import build_model, unfreeze_model
from data import DistractionDataLoader
from utils import plot_training_history, save_confusion_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train driver distraction detection model'
    )
    
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='imgs',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--model-path', 
        type=str, 
        default='models/distraction_model.h5',
        help='Path to save the trained model'
    )
    
    parser.add_argument(
        '--backbone', 
        type=str, 
        default='efficientnetb0',
        choices=['efficientnetb0', 'mobilenetv2', 'custom'],
        help='Backbone model architecture'
    )
    
    parser.add_argument(
        '--img-size', 
        type=int, 
        default=224,
        help='Input image size (assumes square images)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=1e-4,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--dropout-rate', 
        type=float, 
        default=0.5,
        help='Dropout rate for regularization'
    )
    
    parser.add_argument(
        '--test-split', 
        type=float, 
        default=0.2,
        help='Fraction of data to use for testing'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=7,
        help='Early stopping patience'
    )
    
    parser.add_argument(
        '--reduce-lr-patience', 
        type=int, 
        default=3,
        help='Reduce learning rate patience'
    )
    
    parser.add_argument(
        '--fine-tune', 
        action='store_true',
        help='Fine-tune the base model after initial training'
    )
    
    parser.add_argument(
        '--fine-tune-epochs', 
        type=int, 
        default=20,
        help='Number of fine-tuning epochs'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Directory to save outputs (logs, plots, etc.)'
    )
    
    parser.add_argument(
        '--use-augmentation', 
        action='store_true',
        help='Use data augmentation during training'
    )
    
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory and subdirectories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    return output_dir

def create_callbacks(model_path, output_dir, early_stopping_patience, reduce_lr_patience):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    callbacks.append(tensorboard)
    
    # CSV logger
    csv_logger = CSVLogger(
        os.path.join(output_dir, 'logs', 'training_log.csv'),
        append=True
    )
    callbacks.append(csv_logger)
    
    return callbacks

def train_model(model, train_data, val_data, callbacks, epochs, verbose=1):
    """Train the model."""
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=32,  # Will be overridden if using generators
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True
    )
    
    return history

def fine_tune_model(model, train_data, val_data, callbacks, epochs, verbose=1):
    """Fine-tune the model by unfreezing some layers."""
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    
    # Unfreeze the top layers of the base model
    model = unfreeze_model(model, unfreeze_layers=30)
    
    # Train with a lower learning rate
    history = model.fit(
        train_data[0], train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True
    )
    
    return history

def save_training_config(args, output_dir):
    """Save training configuration to JSON file."""
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to {config_path}")

def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Save training configuration
    save_training_config(args, output_dir)
    
    print("Driver Distraction Detection Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Backbone: {args.backbone}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\nLoading data...")
    data_loader = DistractionDataLoader(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    try:
        X_train, X_test, y_train, y_test, class_names = data_loader.load_data_from_folders(
            test_size=args.test_split,
            random_state=args.random_state
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have images in the correct folder structure.")
        return
    
    # Create data generators if augmentation is enabled
    if args.use_augmentation:
        print("Using data augmentation...")
        train_generator, test_generator = data_loader.create_data_generators(
            X_train, y_train, X_test, y_test
        )
        train_data = train_generator
        val_data = (X_test, y_test)
    else:
        train_data = (X_train, y_train)
        val_data = (X_test, y_test)
    
    # Plot class distribution
    data_loader.plot_class_distribution(
        y_train, y_test, 
        save_path=os.path.join(output_dir, 'plots', 'class_distribution.png')
    )
    
    # Build model
    print(f"\nBuilding {args.backbone} model...")
    model = build_model(
        num_classes=len(class_names),
        input_shape=(args.img_size, args.img_size, 3),
        backbone=args.backbone,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Create callbacks
    model_path = os.path.join(output_dir, 'models', 'distraction_model.h5')
    callbacks = create_callbacks(
        model_path, output_dir, 
        args.early_stopping_patience, 
        args.reduce_lr_patience
    )
    
    # Train model
    history = train_model(
        model, train_data, val_data, callbacks, args.epochs
    )
    
    # Fine-tune if requested
    if args.fine_tune:
        print("\nFine-tuning model...")
        fine_tune_history = fine_tune_model(
            model, train_data, val_data, callbacks, args.fine_tune_epochs
        )
        
        # Combine histories
        for key in history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_top3_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test top-3 accuracy: {test_top3_accuracy:.4f}")
    
    # Generate predictions for confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Save confusion matrix
    cm_path = os.path.join(output_dir, 'plots', 'confusion_matrix.png')
    save_confusion_matrix(
        y_true_classes, y_pred_classes, class_names, cm_path
    )
    
    # Plot training history
    history_path = os.path.join(output_dir, 'plots', 'training_history.png')
    plot_training_history(history, history_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'models', 'final_distraction_model.h5')
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Training history saved to: {history_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 