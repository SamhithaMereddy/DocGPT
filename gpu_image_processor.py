"""
GPU-Accelerated Image Processing for Document Analysis
Provides GPU tensor operations for faster image preprocessing
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
import logging
from typing import List, Tuple, Optional, Union
from gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

class GPUImageProcessor:
    """GPU-accelerated image processing for document analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Setup GPU optimizations
        if torch.cuda.is_available():
            gpu_optimizer.setup_mixed_precision()
            
        # Pre-compiled transforms for speed
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logger.info(f"GPU Image Processor initialized on {self.device}")
    
    def preprocess_image_batch(self, images: List[Image.Image], 
                              target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        GPU-accelerated batch image preprocessing
        
        Args:
            images: List of PIL Images
            target_size: Target size (height, width)
            
        Returns:
            GPU tensor batch ready for model inference
        """
        try:
            # Convert PIL images to tensor batch
            tensor_list = []
            
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to tensor and normalize to [0,1]
                tensor = TF.to_tensor(img)
                tensor_list.append(tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(tensor_list).to(self.device, dtype=self.dtype)
            
            # GPU-accelerated resize using tensor operations
            if target_size != batch_tensor.shape[-2:]:
                batch_tensor = F.interpolate(
                    batch_tensor,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                    antialias=True
                )
            
            # Apply normalization on GPU
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                batch_tensor = self.normalize_transform(batch_tensor)
            
            return batch_tensor
            
        except Exception as e:
            logger.error(f"Error in GPU image preprocessing: {e}")
            raise
    
    def enhance_image_contrast(self, image_tensor: torch.Tensor, 
                              contrast_factor: float = 1.5) -> torch.Tensor:
        """GPU-accelerated contrast enhancement"""
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Calculate mean for contrast adjustment
                mean = torch.mean(image_tensor, dim=(-2, -1), keepdim=True)
                enhanced = (image_tensor - mean) * contrast_factor + mean
                return torch.clamp(enhanced, 0, 1)
                
        except Exception as e:
            logger.error(f"Error in GPU contrast enhancement: {e}")
            return image_tensor
    
    def apply_gaussian_blur(self, image_tensor: torch.Tensor, 
                           kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """GPU-accelerated Gaussian blur using conv2d"""
        try:
            # Create Gaussian kernel on GPU
            kernel = self._create_gaussian_kernel(kernel_size, sigma)
            kernel = kernel.to(self.device, dtype=self.dtype)
            
            # Apply convolution for each channel
            channels = image_tensor.shape[1]
            kernel = kernel.repeat(channels, 1, 1, 1)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                blurred = F.conv2d(
                    image_tensor,
                    kernel,
                    padding=kernel_size // 2,
                    groups=channels  # Depthwise convolution
                )
            
            return blurred
            
        except Exception as e:
            logger.error(f"Error in GPU Gaussian blur: {e}")
            return image_tensor
    
    def detect_edges_gpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated edge detection using Sobel filters"""
        try:
            # Sobel kernels for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                 dtype=self.dtype, device=self.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                 dtype=self.dtype, device=self.device)
            
            # Reshape for conv2d
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            # Convert to grayscale if RGB
            if image_tensor.shape[1] == 3:
                gray = torch.mean(image_tensor, dim=1, keepdim=True)
            else:
                gray = image_tensor
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Apply Sobel filters
                edge_x = F.conv2d(gray, sobel_x, padding=1)
                edge_y = F.conv2d(gray, sobel_y, padding=1)
                
                # Combine edges
                edges = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            
            return edges
            
        except Exception as e:
            logger.error(f"Error in GPU edge detection: {e}")
            return image_tensor
    
    def resize_batch_gpu(self, image_tensor: torch.Tensor, 
                        scale_factor: float = 2.0) -> torch.Tensor:
        """GPU-accelerated batch image resizing"""
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                resized = F.interpolate(
                    image_tensor,
                    scale_factor=scale_factor,
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                )
            return resized
            
        except Exception as e:
            logger.error(f"Error in GPU resize: {e}")
            return image_tensor
    
    def adaptive_threshold_gpu(self, image_tensor: torch.Tensor, 
                              block_size: int = 11, C: float = 2.0) -> torch.Tensor:
        """GPU-accelerated adaptive thresholding"""
        try:
            # Convert to grayscale if needed
            if image_tensor.shape[1] == 3:
                gray = torch.mean(image_tensor, dim=1, keepdim=True)
            else:
                gray = image_tensor
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Create averaging kernel
                kernel_size = block_size
                kernel = torch.ones(1, 1, kernel_size, kernel_size, 
                                  dtype=self.dtype, device=self.device) / (kernel_size ** 2)
                
                # Compute local mean
                local_mean = F.conv2d(gray, kernel, padding=kernel_size // 2)
                
                # Apply adaptive threshold
                threshold = local_mean - C / 255.0
                binary = (gray > threshold).float()
            
            return binary
            
        except Exception as e:
            logger.error(f"Error in GPU adaptive threshold: {e}")
            return image_tensor
    
    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian kernel for blur operations"""
        coords = torch.arange(kernel_size, dtype=self.dtype) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel_2d = g[:, None] * g[None, :]
        return kernel_2d.view(1, 1, kernel_size, kernel_size)
    
    def batch_to_pil(self, tensor_batch: torch.Tensor) -> List[Image.Image]:
        """Convert GPU tensor batch back to PIL Images"""
        try:
            # Move to CPU and denormalize
            tensor_batch = tensor_batch.cpu()
            
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor_batch = tensor_batch * std + mean
            tensor_batch = torch.clamp(tensor_batch, 0, 1)
            
            # Convert to PIL
            images = []
            for tensor in tensor_batch:
                pil_image = TF.to_pil_image(tensor)
                images.append(pil_image)
            
            return images
            
        except Exception as e:
            logger.error(f"Error converting tensor to PIL: {e}")
            return []
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        return gpu_optimizer.get_gpu_memory_info()
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        gpu_optimizer.cleanup_gpu_memory(force=True)

# Global GPU image processor instance
gpu_image_processor = GPUImageProcessor()