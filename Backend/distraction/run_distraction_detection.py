#!/usr/bin/env python3
"""
Driver Distraction Detection - Unified Interface
Keras-based deep learning system for real-time driver distraction detection
"""

import os
import sys
import argparse
import subprocess

def print_banner():
    """Print system banner."""
    print("=" * 60)
    print("🚗 DRIVER DISTRACTION DETECTION SYSTEM")
    print("   Keras + TensorFlow 2.x Implementation")
    print("=" * 60)

def print_help():
    """Print detailed help information."""
    print("\n📋 Available Commands:")
    print("  train     - Train a new distraction detection model")
    print("  inference - Run real-time inference on video stream")
    print("  test      - Test system components")
    print("  help      - Show this help message")
    
    print("\n🚀 Quick Start Examples:")
    print("  # Train a model with default settings")
    print("  python run_distraction_detection.py train --data-dir imgs")
    
    print("  # Run real-time detection from IP camera")
    print("  python run_distraction_detection.py inference --model-path outputs/models/distraction_model.h5")
    
    print("  # Test system components")
    print("  python run_distraction_detection.py test")
    
    print("\n📊 Training Options:")
    print("  --data-dir              Directory containing training data")
    print("  --backbone              Model backbone (efficientnetb0, mobilenetv2, custom)")
    print("  --epochs                Number of training epochs")
    print("  --batch-size            Training batch size")
    print("  --learning-rate         Initial learning rate")
    print("  --use-augmentation      Enable data augmentation")
    print("  --fine-tune             Enable fine-tuning")
    
    print("\n🎥 Inference Options:")
    print("  --model-path            Path to trained model (.h5 file)")
    print("  --source                Video source (IP camera URL, file path, or camera index)")
    print("  --output                Output video path (optional)")
    print("  --confidence-threshold  Minimum confidence for predictions")
    
    print("\n📁 Data Structure:")
    print("  imgs/")
    print("  ├── c0_safe_driving/")
    print("  ├── c1_texting_right/")
    print("  ├── c2_phone_right/")
    print("  ├── c3_texting_left/")
    print("  ├── c4_phone_left/")
    print("  ├── c5_adjusting_radio/")
    print("  ├── c6_drinking/")
    print("  ├── c7_reaching_behind/")
    print("  ├── c8_hair_makeup/")
    print("  └── c9_talking_passenger/")

def run_training(args):
    """Run the training script."""
    print("🎯 Starting Model Training...")
    
    # Build command
    cmd = ["python", "train.py"]
    
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    if args.backbone:
        cmd.extend(["--backbone", args.backbone])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.use_augmentation:
        cmd.append("--use-augmentation")
    if args.fine_tune:
        cmd.append("--fine-tune")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ Error: train.py not found")
        return False
    
    return True

def run_inference(args):
    """Run the inference script."""
    print("🎥 Starting Real-time Inference...")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("Please train a model first or provide the correct path.")
        return False
    
    # Build command
    cmd = ["python", "inference.py"]
    cmd.extend(["--model-path", args.model_path])
    
    if args.source:
        cmd.extend(["--source", args.source])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.confidence_threshold:
        cmd.extend(["--confidence-threshold", str(args.confidence_threshold)])
    if args.img_size:
        cmd.extend(["--img-size", str(args.img_size)])
    if args.no_logs:
        cmd.append("--no-logs")
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run inference
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ Error: inference.py not found")
        return False
    
    return True

def run_test():
    """Run the system test."""
    print("🧪 Testing System Components...")
    
    cmd = ["python", "test_system.py"]
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed")
            return False
    except FileNotFoundError:
        print("❌ Error: test_system.py not found")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Driver Distraction Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_distraction_detection.py train --data-dir imgs --epochs 50
  python run_distraction_detection.py inference --model-path models/distraction_model.h5
  python run_distraction_detection.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data-dir', default='imgs', help='Training data directory')
    train_parser.add_argument('--backbone', default='efficientnetb0', 
                             choices=['efficientnetb0', 'mobilenetv2', 'custom'],
                             help='Model backbone')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--use-augmentation', action='store_true', help='Enable augmentation')
    train_parser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning')
    train_parser.add_argument('--output-dir', default='outputs', help='Output directory')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run real-time inference')
    inference_parser.add_argument('--model-path', required=True, help='Path to trained model')
    inference_parser.add_argument('--source', default='http://10.56.19.74:8080/video', 
                                 help='Video source (IP camera, file, or camera index)')
    inference_parser.add_argument('--output', help='Output video path')
    inference_parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                                 help='Confidence threshold')
    inference_parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    inference_parser.add_argument('--no-logs', action='store_true', help='Disable logging')
    
    # Test command
    subparsers.add_parser('test', help='Test system components')
    
    # Help command
    subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle commands
    if args.command == 'train':
        success = run_training(args)
        if success:
            print("\n✅ Training completed successfully!")
            print("Next: Run inference with the trained model")
        else:
            print("\n❌ Training failed")
            sys.exit(1)
    
    elif args.command == 'inference':
        success = run_inference(args)
        if not success:
            sys.exit(1)
    
    elif args.command == 'test':
        success = run_test()
        if not success:
            sys.exit(1)
    
    elif args.command == 'help':
        print_help()
    
    else:
        print_help()
        print("\n💡 Tip: Use 'python run_distraction_detection.py help' for detailed information")

if __name__ == "__main__":
    main() 