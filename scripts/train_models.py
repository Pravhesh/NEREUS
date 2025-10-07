"""
Model Training Script
Train YOLOv5n detector and MobileNetV3 classifier
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from detection import YOLODetector
from classification import MobileNetClassifier


def train_detector(data_yaml: str, 
                  epochs: int = 100,
                  batch_size: int = 16,
                  device: str = 'cuda'):
    """
    Train YOLOv5n detector
    
    Args:
        data_yaml: Path to YOLO data.yaml
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
    """
    print("=" * 50)
    print("Training YOLOv5n Detector")
    print("=" * 50)
    
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for detector training, but no CUDA device is available.")

    # Initialize detector
    detector = YOLODetector(device=device)
    
    # Print model info
    info = detector.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Device: {info['device']}")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {data_yaml}")
    
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch_size=batch_size,
        project='runs/detect',
        name='zooplankton_yolov5n'
    )
    
    print("\nTraining completed!")
    print(f"Results saved to: runs/detect/zooplankton_yolov5n")
    
    # Validate
    print("\nValidating model...")
    metrics = detector.validate(data_yaml)
    print(f"Validation metrics:")
    print(f"  mAP@0.5: {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map']:.4f}")
    
    return detector


def train_classifier(data_dir: str,
                    num_classes: int,
                    epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    device: str = 'cuda'):
    """
    Train MobileNetV3 classifier
    
    Args:
        data_dir: Path to classification dataset
        num_classes: Number of classes
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    print("=" * 50)
    print("Training MobileNetV3 Classifier")
    print("=" * 50)
    
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for classifier training, but no CUDA device is available.")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    data_path = Path(data_dir)
    
    train_dataset = datasets.ImageFolder(
        root=data_path / 'train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=data_path / 'val',
        transform=val_transform
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize classifier
    classifier = MobileNetClassifier(
        num_classes=num_classes,
        device=device,
        use_embeddings=True
    )
    
    # Print model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    save_path = 'models/mobilenetv3_classifier_best.pt'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {save_path}")
    
    return classifier


def export_models(detector: YOLODetector, 
                 classifier: MobileNetClassifier,
                 output_dir: str = 'models/tflite'):
    """
    Export models to TFLite for edge deployment
    
    Args:
        detector: Trained detector
        classifier: Trained classifier
        output_dir: Output directory
    """
    print("\n" + "=" * 50)
    print("Exporting Models to TFLite")
    print("=" * 50)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export detector
    print("\nExporting detector...")
    detector.export_to_tflite(
        str(output_path / 'yolov5n_detector.tflite'),
        quantize=True
    )
    
    # Export classifier
    print("\nExporting classifier...")
    classifier.export_to_tflite(
        str(output_path / 'mobilenetv3_classifier.tflite'),
        quantize=True
    )
    
    print(f"\nModels exported to: {output_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train NEREUS models')
    parser.add_argument('--mode', type=str, choices=['detector', 'classifier', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--data-yaml', type=str, 
                       default='prepared_dataset/yolo/data.yaml',
                       help='Path to YOLO data.yaml')
    parser.add_argument('--classification-dir', type=str,
                       default='prepared_dataset/classification',
                       help='Path to classification dataset')
    parser.add_argument('--num-classes', type=int, default=127,
                       help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--export', action='store_true',
                       help='Export models to TFLite after training')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Error: CUDA was requested but is not available on this system. Aborting training.")
        sys.exit(1)
    
    print("=" * 50)
    print("NEREUS Model Training")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    detector = None
    classifier = None
    
    # Train detector
    if args.mode in ['detector', 'both']:
        detector = train_detector(
            data_yaml=args.data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    # Train classifier
    if args.mode in ['classifier', 'both']:
        classifier = train_classifier(
            data_dir=args.classification_dir,
            num_classes=args.num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    # Export models
    if args.export and detector is not None and classifier is not None:
        export_models(detector, classifier)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
