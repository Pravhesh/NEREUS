"""
Dataset Preparation Script
Prepares the zooplankton dataset for training YOLOv5 and MobileNetV3 models
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class DatasetPreparer:
    """Prepares zooplankton dataset for model training"""
    
    def __init__(self, root_dir: str):
        """
        Initialize dataset preparer
        
        Args:
            root_dir: Root directory of the dataset
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / 'individual_images'
        self.taxonomy_file = self.root_dir / 'taxonomy_descriptor_zooscan.csv'
        
        # Output directories
        self.output_dir = self.root_dir / 'prepared_dataset'
        self.yolo_dir = self.output_dir / 'yolo'
        self.classification_dir = self.output_dir / 'classification'
        
        # Class mapping
        self.class_to_id = {}
        self.id_to_class = {}
    
    def load_taxonomy(self):
        """Load taxonomy information"""
        print("Loading taxonomy information...")
        
        # Get all class directories
        class_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()]
        
        # Create class mapping
        for i, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            self.class_to_id[class_name] = i
            self.id_to_class[i] = class_name
        
        print(f"Found {len(self.class_to_id)} classes")
        
        # Save class mapping
        mapping_file = self.output_dir / 'class_mapping.json'
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump({
                'class_to_id': self.class_to_id,
                'id_to_class': self.id_to_class,
                'num_classes': len(self.class_to_id)
            }, f, indent=2)
        
        print(f"Class mapping saved to {mapping_file}")
    
    def prepare_classification_dataset(self, 
                                      train_split: float = 0.7,
                                      val_split: float = 0.15,
                                      test_split: float = 0.15):
        """
        Prepare dataset for classification training
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
        """
        print("\nPreparing classification dataset...")
        
        # Create directories
        for split in ['train', 'val', 'test']:
            split_dir = self.classification_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each class
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for class_name, class_id in tqdm(self.class_to_id.items(), desc="Processing classes"):
            class_dir = self.images_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Get all images
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if len(images) == 0:
                continue
            
            # Shuffle
            np.random.shuffle(images)
            
            # Split
            n_train = int(len(images) * train_split)
            n_val = int(len(images) * val_split)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy images to respective directories
            for split, split_images in [('train', train_images), 
                                       ('val', val_images), 
                                       ('test', test_images)]:
                split_class_dir = self.classification_dir / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in split_images:
                    dst = split_class_dir / img_path.name
                    shutil.copy2(img_path, dst)
                    stats[split] += 1
        
        print(f"\nClassification dataset prepared:")
        print(f"  Train: {stats['train']} images")
        print(f"  Val: {stats['val']} images")
        print(f"  Test: {stats['test']} images")
        print(f"  Location: {self.classification_dir}")
    
    def prepare_detection_dataset(self,
                                  train_split: float = 0.7,
                                  val_split: float = 0.15,
                                  test_split: float = 0.15,
                                  create_composite: bool = True,
                                  images_per_composite: int = 5):
        """
        Prepare dataset for YOLO detection training
        Creates composite images with multiple specimens
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            create_composite: Whether to create composite images
            images_per_composite: Number of specimens per composite
        """
        print("\nPreparing detection dataset...")
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        if create_composite:
            self._create_composite_images(train_split, val_split, test_split, 
                                         images_per_composite)
        else:
            self._create_single_object_dataset(train_split, val_split, test_split)
        
        # Create data.yaml for YOLO
        self._create_yolo_yaml()
    
    def _create_composite_images(self, train_split, val_split, test_split, 
                                images_per_composite):
        """Create composite images with multiple specimens"""
        print("Creating composite images...")
        
        # Collect all images
        all_images = []
        for class_name, class_id in self.class_to_id.items():
            class_dir = self.images_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                for img_path in images:
                    all_images.append((img_path, class_id))
        
        np.random.shuffle(all_images)
        
        # Split
        n_train = int(len(all_images) * train_split)
        n_val = int(len(all_images) * val_split)
        
        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:]
        }
        
        # Create composite images for each split
        for split_name, split_images in splits.items():
            print(f"Creating {split_name} composites...")
            
            # Create composites
            n_composites = len(split_images) // images_per_composite
            
            for i in tqdm(range(n_composites), desc=f"{split_name} composites"):
                # Select random images for this composite
                composite_images = split_images[i * images_per_composite:
                                               (i + 1) * images_per_composite]
                
                # Create composite
                composite_img, annotations = self._create_single_composite(
                    composite_images, canvas_size=640
                )
                
                # Save image
                img_name = f"composite_{i:06d}.jpg"
                img_path = self.yolo_dir / split_name / 'images' / img_name
                cv2.imwrite(str(img_path), composite_img)
                
                # Save annotations in YOLO format
                label_path = self.yolo_dir / split_name / 'labels' / f"composite_{i:06d}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class']} {ann['x_center']} {ann['y_center']} "
                               f"{ann['width']} {ann['height']}\n")
        
        print("Composite images created successfully!")
    
    def _create_single_composite(self, images_with_classes, canvas_size=640):
        """Create a single composite image"""
        # Create blank canvas
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 200
        
        annotations = []
        
        for img_path, class_id in images_with_classes:
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Resize if too large
            max_size = canvas_size // 3
            h, w = img.shape
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
                h, w = new_h, new_w
            
            # Random position
            max_x = canvas_size - w
            max_y = canvas_size - h
            
            if max_x <= 0 or max_y <= 0:
                continue
            
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            
            # Place on canvas
            canvas[y:y+h, x:x+w] = img
            
            # Create YOLO annotation (normalized)
            x_center = (x + w / 2) / canvas_size
            y_center = (y + h / 2) / canvas_size
            width = w / canvas_size
            height = h / canvas_size
            
            annotations.append({
                'class': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        
        return canvas, annotations
    
    def _create_single_object_dataset(self, train_split, val_split, test_split):
        """Create dataset with single objects (simpler approach)"""
        print("Creating single-object detection dataset...")
        
        # Similar to classification but with YOLO annotations
        for class_name, class_id in tqdm(self.class_to_id.items(), desc="Processing"):
            class_dir = self.images_dir / class_name
            
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if len(images) == 0:
                continue
            
            np.random.shuffle(images)
            
            # Split
            n_train = int(len(images) * train_split)
            n_val = int(len(images) * val_split)
            
            splits = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }
            
            for split_name, split_images in splits.items():
                for img_path in split_images:
                    # Copy image
                    dst_img = self.yolo_dir / split_name / 'images' / img_path.name
                    shutil.copy2(img_path, dst_img)
                    
                    # Create annotation (object fills entire image)
                    label_path = self.yolo_dir / split_name / 'labels' / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        # Center of image, full size
                        f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
    
    def _create_yolo_yaml(self):
        """Create data.yaml for YOLO training"""
        yaml_content = f"""# NEREUS Zooplankton Detection Dataset
path: {self.yolo_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(self.class_to_id)}
names: {list(self.id_to_class.values())}
"""
        
        yaml_path = self.yolo_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"YOLO data.yaml created: {yaml_path}")
    
    def get_dataset_statistics(self):
        """Get statistics about the dataset"""
        print("\n" + "=" * 50)
        print("Dataset Statistics")
        print("=" * 50)
        
        stats = {}
        total_images = 0
        
        for class_name in self.class_to_id.keys():
            class_dir = self.images_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                count = len(images)
                stats[class_name] = count
                total_images += count
        
        print(f"\nTotal classes: {len(self.class_to_id)}")
        print(f"Total images: {total_images}")
        print(f"Average images per class: {total_images / len(self.class_to_id):.1f}")
        
        # Top 10 classes
        print("\nTop 10 classes by image count:")
        sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
        for i, (class_name, count) in enumerate(sorted_stats[:10], 1):
            print(f"  {i}. {class_name}: {count} images")
        
        return stats


def main():
    """Main function"""
    print("=" * 50)
    print("NEREUS Dataset Preparation")
    print("=" * 50)
    
    # Initialize preparer
    root_dir = Path(__file__).parent.parent
    preparer = DatasetPreparer(root_dir)
    
    # Load taxonomy
    preparer.load_taxonomy()
    
    # Get statistics
    preparer.get_dataset_statistics()
    
    # Prepare datasets
    print("\n" + "=" * 50)
    print("Preparing datasets...")
    print("=" * 50)
    
    # Classification dataset
    preparer.prepare_classification_dataset(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # Detection dataset (with composite images)
    preparer.prepare_detection_dataset(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        create_composite=True,
        images_per_composite=5
    )
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("=" * 50)
    print(f"\nOutput directory: {preparer.output_dir}")
    print(f"  - Classification: {preparer.classification_dir}")
    print(f"  - Detection (YOLO): {preparer.yolo_dir}")


if __name__ == '__main__':
    main()
