"""
Image processing utilities
Preprocessing, format conversion, validation
"""

import numpy as np
from PIL import Image
import logging

from config import settings

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handle image preprocessing and conversion"""
    
    def __init__(self):
        self.target_size = settings.IMAGE_SIZE
        logger.info(f"ImageProcessor initialized with target size: {self.target_size}")
    
    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed (maintaining aspect ratio)
        if image.size[0] != self.target_size or image.size[1] != self.target_size:
            image = self._resize_with_padding(image, self.target_size)
        
        return image
    
    def _resize_with_padding(self, image: Image.Image, target_size: int) -> Image.Image:
        """
        Resize image maintaining aspect ratio with padding
        
        Args:
            image: PIL Image
            target_size: Target size for both dimensions
            
        Returns:
            Resized image with padding
        """
        # Calculate aspect ratio
        aspect_ratio = image.width / image.height
        
        if aspect_ratio > 1:
            # Wider than tall
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            # Taller than wide
            new_height = target_size
            new_width = int(target_size * aspect_ratio)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        
        # Paste resized image in center
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        new_image.paste(resized, (paste_x, paste_y))
        
        return new_image
    
    def array_to_pil(self, array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image
        
        Args:
            array: Numpy array (RGB format)
            
        Returns:
            PIL Image
        """
        # Ensure uint8
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
        
        return Image.fromarray(array)
    
    def pil_to_array(self, image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array
        
        Args:
            image: PIL Image
            
        Returns:
            Numpy array
        """
        return np.array(image)
    
    def validate_image(self, image: Image.Image) -> bool:
        """
        Validate image format and size
        
        Args:
            image: PIL Image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check dimensions
            if image.width < 64 or image.height < 64:
                logger.warning(f"Image too small: {image.size}")
                return False
            
            if image.width > 4096 or image.height > 4096:
                logger.warning(f"Image too large: {image.size}")
                return False
            
            # Check mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                logger.warning(f"Unsupported image mode: {image.mode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False