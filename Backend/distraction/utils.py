#!/usr/bin/env python3
"""
Utility functions for driver distraction detection
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import cv2
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

def plot_training_history(history, save_path=None, show_plot=True):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
        show_plot: Whether to display the plot (default: True)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training & Validation Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training & Validation Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Top-3 Accuracy (if available)
    if 'top_3_accuracy' in history.history:
        axes[1, 1].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        axes[1, 1].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        axes[1, 1].set_title('Top-3 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-3 Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    if show_plot:
        plt.show()

def save_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=True, show_plot=True):
    """
    Create and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
        show_plot: Whether to display the plot (default: True)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return cm

def print_classification_metrics(y_true, y_pred, class_names):
    """
    Print detailed classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 60)
    
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20s}: Precision={precision[i]:.3f}, "
              f"Recall={recall[i]:.3f}, F1={f1[i]:.3f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

def create_gradcam_visualization(model, image, class_index, layer_name=None):
    """
    Create Grad-CAM visualization for model interpretability.
    
    Args:
        model: Trained Keras model
        image: Input image (preprocessed)
        class_index: Index of the class to visualize
        layer_name: Name of the layer to use for Grad-CAM (default: last conv layer)
        
    Returns:
        Grad-CAM heatmap
    """
    # Get the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Convolutional layer
                layer_name = layer.name
                break
    
    # Create a model that outputs the last conv layer and the predictions
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_index]
    
    # Extract the gradients of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(loss, conv_outputs)
    
    # Vector-mean of the gradients over the batch dimension
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by corresponding gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on the original image.
    
    Args:
        image: Original image (RGB format)
        heatmap: Grad-CAM heatmap
        alpha: Transparency factor
        
    Returns:
        Image with overlaid heatmap
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert image to RGB if it's in BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Overlay heatmap on image
    output = heatmap * alpha + image_rgb * (1 - alpha)
    output = output / output.max()
    
    return output

def save_gradcam_visualization(model, image_path, class_names, save_dir='gradcam_outputs'):
    """
    Create and save Grad-CAM visualizations for all classes.
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        class_names: List of class names
        save_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Preprocess for model
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Get model predictions
    predictions = model.predict(image_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    
    print(f"Predicted class: {class_names[predicted_class]} ({predictions[0][predicted_class]:.3f})")
    
    # Create visualizations for top 3 predictions
    top_indices = np.argsort(predictions[0])[-3:][::-1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Top 3 predictions
    for i, class_idx in enumerate(top_indices):
        # Create Grad-CAM
        heatmap = create_gradcam_visualization(model, image_batch, class_idx)
        
        # Overlay on image
        overlay = overlay_heatmap_on_image(image_rgb, heatmap)
        
        # Plot
        row = i // 2
        col = (i % 2) + 1
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f'{class_names[class_idx]}: {predictions[0][class_idx]:.3f}')
        axes[row, col].axis('off')
    
    # Hide unused subplot
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'gradcam_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to {save_path}")
    
    plt.show()

def plot_class_probabilities(probabilities, class_names, save_path=None):
    """
    Plot class probabilities as a bar chart.
    
    Args:
        probabilities: Array of class probabilities
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(class_names)), probabilities)
    
    # Color bars based on probability
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        if prob > 0.7:
            bar.set_color('green')
        elif prob > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.title('Class Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, prob in enumerate(probabilities):
        plt.text(i, prob + 0.01, f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class probabilities plot saved to {save_path}")
    
    plt.show()

def create_model_summary_plot(model, save_path=None):
    """
    Create a visual summary of the model architecture.
    
    Args:
        model: Keras model
        save_path: Path to save the plot
    """
    # Get model summary as string
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_str = '\n'.join(summary_list)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(0.05, 0.95, summary_str, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax.set_title('Model Architecture Summary')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model summary plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test utility functions
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("- plot_training_history()")
    print("- save_confusion_matrix()")
    print("- print_classification_metrics()")
    print("- create_gradcam_visualization()")
    print("- overlay_heatmap_on_image()")
    print("- save_gradcam_visualization()")
    print("- plot_class_probabilities()")
    print("- create_model_summary_plot()") 