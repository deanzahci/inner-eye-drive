import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

class DistractionDataLoader:
    """
    Data loader for driver distraction detection dataset.
    """
    
    def __init__(self, data_dir='imgs', img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = [
            'c0_safe_driving',
            'c1_texting_right',
            'c2_phone_right', 
            'c3_texting_left',
            'c4_phone_left',
            'c5_adjusting_radio',
            'c6_drinking',
            'c7_reaching_behind',
            'c8_hair_makeup',
            'c9_talking_passenger'
        ]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
    def load_data_from_folders(self, test_size=0.2, random_state=42):
        """
        Load data from folder structure where each class has its own folder.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, class_names)
        """
        images = []
        labels = []
        
        print("Loading images from folders...")
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found, skipping...")
                continue
                
            # Get all image files in the class directory
            image_files = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                         glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(class_dir, "*.png"))
            
            print(f"Found {len(image_files)} images in {class_name}")
            
            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        if not images:
            raise ValueError("No images found in the data directory!")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images with {len(np.unique(y))} classes")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Convert to categorical
        y_train = to_categorical(y_train, num_classes=len(self.class_names))
        y_test = to_categorical(y_test, num_classes=len(self.class_names))
        
        return X_train, X_test, y_train, y_test, self.class_names
    
    def load_data_from_csv(self, csv_path, img_col='img', label_col='classname', 
                          test_size=0.2, random_state=42):
        """
        Load data from CSV file with image paths and labels.
        
        Args:
            csv_path: Path to CSV file
            img_col: Column name containing image paths
            label_col: Column name containing class labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, class_names)
        """
        df = pd.read_csv(csv_path)
        
        images = []
        labels = []
        
        print("Loading images from CSV...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                img_path = row[img_col]
                label = row[label_col]
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"Error loading image at row {idx}: {e}")
                continue
        
        if not images:
            raise ValueError("No images found in the CSV file!")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        print(f"Loaded {len(X)} images with {len(np.unique(y_encoded))} classes")
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, 
            stratify=y_encoded
        )
        
        # Convert to categorical
        y_train = to_categorical(y_train, num_classes=len(self.class_names))
        y_test = to_categorical(y_test, num_classes=len(self.class_names))
        
        return X_train, X_test, y_train, y_test, self.class_names
    
    def create_data_generators(self, X_train, y_train, X_test, y_test):
        """
        Create data generators with augmentation for training.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Tuple of (train_generator, test_generator)
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Test data generator (no augmentation)
        test_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_generator = test_datagen.flow(
            X_test, y_test,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, test_generator
    
    def get_class_distribution(self, y):
        """
        Get class distribution from labels.
        
        Args:
            y: Categorical labels
            
        Returns:
            Dictionary with class counts
        """
        class_counts = np.sum(y, axis=0)
        return dict(zip(self.class_names, class_counts))
    
    def plot_class_distribution(self, y_train, y_test, save_path=None):
        """
        Plot class distribution for training and test sets.
        
        Args:
            y_train: Training labels
            y_test: Test labels
            save_path: Path to save the plot
        """
        train_counts = self.get_class_distribution(y_train)
        test_counts = self.get_class_distribution(y_test)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set distribution
        ax1.bar(range(len(self.class_names)), list(train_counts.values()))
        ax1.set_title('Training Set Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels([name.split('_')[1] for name in self.class_names], rotation=45)
        
        # Test set distribution
        ax2.bar(range(len(self.class_names)), list(test_counts.values()))
        ax2.set_title('Test Set Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels([name.split('_')[1] for name in self.class_names], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        return img

def create_sample_data():
    """
    Create sample data structure for testing if no real data is available.
    """
    # Create sample directory structure
    base_dir = 'imgs'
    os.makedirs(base_dir, exist_ok=True)
    
    class_names = [
        'c0_safe_driving',
        'c1_texting_right',
        'c2_phone_right', 
        'c3_texting_left',
        'c4_phone_left',
        'c5_adjusting_radio',
        'c6_drinking',
        'c7_reaching_behind',
        'c8_hair_makeup',
        'c9_talking_passenger'
    ]
    
    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")

if __name__ == "__main__":
    # Test data loader
    data_loader = DistractionDataLoader()
    
    # Create sample data structure if needed
    if not os.path.exists('imgs'):
        create_sample_data()
        print("Created sample data structure. Please add images to the class folders.")
    else:
        try:
            # Try to load data
            X_train, X_test, y_train, y_test, class_names = data_loader.load_data_from_folders()
            print(f"Successfully loaded data:")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Classes: {class_names}")
            
            # Plot class distribution
            data_loader.plot_class_distribution(y_train, y_test, 'class_distribution.png')
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure you have images in the correct folder structure.") 