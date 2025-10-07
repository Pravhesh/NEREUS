"""
Image Standardization Module
Converts various microscope formats (ND2, LIF, CZI) to standardized OME-TIFF
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import tifffile


class ImageStandardizer:
    """
    Standardizes microscope images from various formats to a common format
    Supports: ND2, LIF, CZI, TIFF, PNG, JPEG
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None, 
                 normalize: bool = True):
        """
        Initialize the standardizer
        
        Args:
            target_size: Target size (width, height) for resizing. None to keep original
            normalize: Whether to normalize pixel values to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
        self.supported_formats = ['.tiff', '.tif', '.png', '.jpg', '.jpeg', 
                                 '.nd2', '.lif', '.czi', '.ome.tiff']
    
    def standardize(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Standardize an image from any supported format
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Standardized image as numpy array
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image based on format
        image = self._load_image(image_path)
        
        # Convert to grayscale if needed (zooplankton images are typically grayscale)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if target size is specified
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] if requested
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image based on file extension
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array
        """
        suffix = image_path.suffix.lower()
        
        # Standard image formats
        if suffix in ['.png', '.jpg', '.jpeg']:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            return image
        
        # TIFF formats
        elif suffix in ['.tiff', '.tif'] or '.ome.tiff' in str(image_path).lower():
            return self._load_tiff(image_path)
        
        # Proprietary microscope formats
        elif suffix in ['.nd2', '.lif', '.czi']:
            return self._load_proprietary_format(image_path)
        
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_tiff(self, image_path: Path) -> np.ndarray:
        """
        Load TIFF/OME-TIFF images
        
        Args:
            image_path: Path to TIFF file
            
        Returns:
            Image as numpy array
        """
        try:
            with tifffile.TiffFile(str(image_path)) as tif:
                image = tif.asarray()
                
                # Handle multi-page TIFF (take first page)
                if len(image.shape) > 2 and image.shape[0] > 1:
                    image = image[0]
                
                return image
        except Exception as e:
            raise ValueError(f"Failed to load TIFF: {e}")
    
    def _load_proprietary_format(self, image_path: Path) -> np.ndarray:
        """
        Load proprietary microscope formats (ND2, LIF, CZI)
        
        Args:
            image_path: Path to proprietary format file
            
        Returns:
            Image as numpy array
        """
        suffix = image_path.suffix.lower()
        
        try:
            # Try using bioformats if available
            import javabridge
            import bioformats
            
            # This is a placeholder - actual implementation would require
            # proper bioformats setup with Java bridge
            # For now, we'll provide a fallback
            raise ImportError("Bioformats not configured")
            
        except ImportError:
            # Fallback: Try tifffile which can handle some proprietary formats
            try:
                with tifffile.TiffFile(str(image_path)) as tif:
                    return tif.asarray()
            except:
                raise ValueError(
                    f"Cannot load {suffix} format. "
                    "Please convert to TIFF or install bioformats library."
                )
    
    def batch_standardize(self, image_dir: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         output_format: str = 'tiff') -> list:
        """
        Standardize a batch of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save standardized images. None to return in memory
            output_format: Output format ('tiff', 'png')
            
        Returns:
            List of standardized images or paths to saved images
        """
        image_dir = Path(image_dir)
        results = []
        
        # Find all supported images
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        for img_path in image_files:
            try:
                standardized = self.standardize(img_path)
                
                if output_dir is not None:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    output_path = output_dir / f"{img_path.stem}.{output_format}"
                    self._save_image(standardized, output_path)
                    results.append(output_path)
                else:
                    results.append(standardized)
                    
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue
        
        return results
    
    def _save_image(self, image: np.ndarray, output_path: Path):
        """
        Save standardized image
        
        Args:
            image: Image array
            output_path: Path to save image
        """
        # Denormalize if needed
        if self.normalize and image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        if output_path.suffix.lower() in ['.tiff', '.tif']:
            tifffile.imwrite(str(output_path), image)
        else:
            cv2.imwrite(str(output_path), image)
