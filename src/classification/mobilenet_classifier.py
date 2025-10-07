"""
MobileNetV3 Classifier Module
Lightweight species classification for zooplankton
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from torchvision import models, transforms
from PIL import Image


class CLAHETransform:
    """
    Custom transform to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for enhanced contrast in microscopy images
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """
        Apply CLAHE to PIL Image
        
        Args:
            img: PIL Image
            
        Returns:
            PIL Image with CLAHE applied
        """
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Check if image is RGB or grayscale
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # For RGB images, apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            limg = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images, apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(img_np)
        
        # Convert back to PIL Image
        return Image.fromarray(enhanced)


class MobileNetClassifier:
    """
    MobileNetV3-based classifier for zooplankton species identification
    Optimized for edge deployment on Raspberry Pi
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 num_classes: int = 127,
                 input_size: int = 224,
                 device: str = 'cpu',
                 use_embeddings: bool = True):
        """
        Initialize MobileNetV3 classifier
        
        Args:
            model_path: Path to trained model weights
            num_classes: Number of species/genus classes
            input_size: Input image size
            device: Device to run inference on
            use_embeddings: Whether to use embedding-based classification
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.use_embeddings = use_embeddings
        
        # Initialize model
        self.model = self._build_model()
        
        # Load weights if provided
        if model_path is not None and Path(model_path).exists():
            self._load_weights(model_path)
        
        self.model.to(device)
        self.model.eval()
        
        # Define preprocessing transforms with CLAHE
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),  # Apply CLAHE
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Store the model's expected input size
        self.input_size = (input_size, input_size, 3) if input_size else (224, 224, 3)
        
        # Class names mapping
        self.class_names = {}
    
    def _build_model(self) -> nn.Module:
        """
        Build MobileNetV3 model with custom classifier head
        """
        # Load pretrained MobileNetV3-Small with default weights
        model = models.mobilenet_v3_small(weights='DEFAULT')
        
        # The classifier in MobileNetV3 is a Sequential with:
        # [0] Linear(576 -> 1024)
        # [1] Hardswish
        # [2] Dropout
        # [3] Linear(1024 -> 1000)
        # We need to replace the entire classifier to match our num_classes
        
        # Get the input features from the feature extractor output
        # MobileNetV3-Small outputs 576 features after avgpool
        in_features = 576
        
        # Create new classifier head
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, self.num_classes)
        )
        
        return model
    
    def _load_weights(self, model_path: str):
        """
        Load model weights
        
        Args:
            model_path: Path to weights file
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")
    
    def classify(self, image: np.ndarray, 
                top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Classify a single image
        
        Args:
            image: Input image (BGR or grayscale)
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_id, confidence) tuples
        """
        # Preprocess image
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append((int(idx.cpu()), float(prob.cpu())))
        
        return results
    
    def classify_batch(self, images: List[np.ndarray], 
                      batch_size: int = 16,
                      top_k: int = 5) -> List[List[Tuple[int, float]]]:
        """
        Classify batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size for inference
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction lists for each image
        """
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(img_rgb)
                batch_tensors.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get top-k for each image
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            for j in range(len(batch)):
                predictions = []
                for prob, idx in zip(top_probs[j], top_indices[j]):
                    predictions.append((int(idx.cpu()), float(prob.cpu())))
                all_predictions.append(predictions)
        
        return all_predictions
    
    def extract_embeddings(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embeddings from image
        
        Args:
            image: Input image
            
        Returns:
            Feature embedding vector
        """
        # Preprocess
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract features from penultimate layer
        with torch.no_grad():
            # Get features before final classification layer
            features = self.model.features(input_tensor)
            features = self.model.avgpool(features)
            embeddings = torch.flatten(features, 1)
            
            # If using embedding-based classifier, get embeddings from embedding layer
            if self.use_embeddings:
                embeddings = self.model.classifier[:4](embeddings)  # Up to embedding layer
        
        return embeddings.cpu().numpy().flatten()
    
    def extract_embeddings_batch(self, images: List[np.ndarray],
                                 batch_size: int = 16) -> np.ndarray:
        """
        Extract embeddings for batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size
            
        Returns:
            Array of embeddings (N x embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(img_rgb)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                features = self.model.features(batch_tensor)
                features = self.model.avgpool(features)
                embeddings = torch.flatten(features, 1)
                
                if self.use_embeddings:
                    embeddings = self.model.classifier[:4](embeddings)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def set_class_names(self, class_names: Dict[int, str]):
        """
        Set class ID to name mapping
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get class name from ID
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name
        """
        return self.class_names.get(class_id, f"Class_{class_id}")
    
    def classify_with_names(self, image: np.ndarray,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify image and return results with class names
        
        Args:
            image: Input image
            top_k: Number of top predictions
            
        Returns:
            List of (class_name, confidence) tuples
        """
        predictions = self.classify(image, top_k)
        
        results = []
        for class_id, conf in predictions:
            class_name = self.get_class_name(class_id)
            results.append((class_name, conf))
        
        return results
    
    def train(self,
             train_loader,
             val_loader,
             epochs: int = 50,
             learning_rate: float = 0.001,
             save_path: Optional[str] = None):
        """
        Train the classifier
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save best model
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
            
            self.model.train()
        
        # Load best model
        if save_path and Path(save_path).exists():
            self._load_weights(save_path)
        
        self.model.eval()
    
    def export_to_tflite(self, output_path: str, quantize: bool = True):
        """
        Export model to TensorFlow Lite
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to apply INT8 quantization
        """
        import torch.onnx
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to ONNX first
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        onnx_path = output_path.with_suffix('.onnx')
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output']
        )
        
        print(f"Model exported to ONNX: {onnx_path}")
        print("Note: Convert ONNX to TFLite using tf2onnx or onnx-tensorflow")
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'MobileNetV3-Small',
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'device': self.device,
            'use_embeddings': self.use_embeddings,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Approximate size in MB
        }
        
        return info
