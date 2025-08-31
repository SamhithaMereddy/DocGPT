"""
GPU Memory Optimizer for Document Search System
Implements model caching, memory management, and batch processing optimizations
"""
import torch
import gc
import logging
from typing import Dict, Any, Optional
from functools import lru_cache
import threading
import time

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """Manages GPU memory and model loading for optimal performance"""
    
    def __init__(self):
        self.model_cache = {}
        self.cache_lock = threading.Lock()
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
        
    def get_gpu_memory_info(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - memory_reserved
            
            return {
                "allocated": memory_allocated,
                "reserved": memory_reserved,
                "free": memory_free,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        return None
    
    def cleanup_gpu_memory(self, force=False):
        """Clean up GPU memory"""
        current_time = time.time()
        
        if force or (current_time - self.last_cleanup) > self.cleanup_interval:
            logger.info("Performing GPU memory cleanup...")
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            self.last_cleanup = current_time
            
            # Log memory status
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                logger.info(f"GPU Memory after cleanup - Allocated: {memory_info['allocated']:.2f}GB, "
                           f"Reserved: {memory_info['reserved']:.2f}GB, Free: {memory_info['free']:.2f}GB")
    
    def get_or_create_model(self, model_type: str, model_path: str, create_func, *args, **kwargs):
        """Get model from cache or create new one with memory optimization"""
        cache_key = f"{model_type}_{model_path}"
        
        with self.cache_lock:
            if cache_key in self.model_cache:
                logger.info(f"Using cached {model_type} model")
                return self.model_cache[cache_key]
            
            # Check memory before loading
            memory_info = self.get_gpu_memory_info()
            if memory_info and memory_info['free'] < 2.0:  # Less than 2GB free
                logger.warning("Low GPU memory, performing cleanup before model loading")
                self.cleanup_gpu_memory(force=True)
            
            # Create new model
            logger.info(f"Creating new {model_type} model")
            model = create_func(*args, **kwargs)
            
            # Cache the model
            self.model_cache[cache_key] = model
            
            logger.info(f"Cached {model_type} model successfully")
            return model
    
    def get_cached_model(self, model_type: str, model_path: str):
        """Get a cached model if it exists"""
        cache_key = f"{model_type}_{model_path}"
        return self.model_cache.get(cache_key)
    
    def optimize_batch_size(self, base_batch_size: int = 32) -> int:
        """Dynamically adjust batch size based on available GPU memory"""
        memory_info = self.get_gpu_memory_info()
        
        if not memory_info:
            return base_batch_size
        
        free_memory_gb = memory_info['free']
        
        # Adjust batch size based on available memory
        if free_memory_gb > 10:
            return min(base_batch_size * 2, 128)
        elif free_memory_gb > 5:
            return base_batch_size
        elif free_memory_gb > 2:
            return max(base_batch_size // 2, 8)
        else:
            return max(base_batch_size // 4, 4)
    
    def enable_memory_efficient_attention(self):
        """Enable memory efficient attention mechanisms"""
        if torch.cuda.is_available():
            try:
                # Enable memory efficient attention
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                logger.info("Enabled memory efficient attention mechanisms")
            except Exception as e:
                logger.warning(f"Could not enable memory efficient attention: {e}")
    
    def setup_mixed_precision(self):
        """Setup mixed precision training for memory efficiency"""
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled mixed precision optimizations")
            except Exception as e:
                logger.warning(f"Could not enable mixed precision: {e}")
    
    def monitor_memory_usage(self, threshold_gb: float = 20.0):
        """Monitor memory usage and trigger cleanup if needed"""
        memory_info = self.get_gpu_memory_info()
        
        if memory_info:
            allocated = memory_info['allocated']
            free = memory_info['free']
            usage_percent = (allocated / memory_info['total']) * 100
            
            # Log current usage
            logger.debug(f"GPU Memory: {allocated:.2f}GB allocated ({usage_percent:.1f}%), {free:.2f}GB free")
            
            # Trigger cleanup based on multiple conditions
            should_cleanup = False
            
            if allocated > threshold_gb:
                logger.warning(f"High GPU memory usage: {allocated:.2f}GB allocated (>{threshold_gb}GB threshold)")
                should_cleanup = True
            elif free < 2.0:  # Less than 2GB free
                logger.warning(f"Low GPU memory available: {free:.2f}GB free")
                should_cleanup = True
            elif usage_percent > 85:  # More than 85% used
                logger.warning(f"High GPU memory usage: {usage_percent:.1f}% used")
                should_cleanup = True
            
            if should_cleanup:
                self.cleanup_gpu_memory(force=True)
                
                # Log after cleanup
                new_memory_info = self.get_gpu_memory_info()
                if new_memory_info:
                    logger.info(f"After cleanup - GPU Memory: {new_memory_info['allocated']:.2f}GB allocated, {new_memory_info['free']:.2f}GB free")
                
                return True
        
        return False
    
    def get_memory_stats(self):
        """Get detailed memory statistics"""
        if not torch.cuda.is_available():
            return None
        
        stats = {
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_memory_allocated': torch.cuda.max_memory_allocated() / 1024**3,
            'max_memory_reserved': torch.cuda.max_memory_reserved() / 1024**3,
            'memory_summary': torch.cuda.memory_summary()
        }
        
        return stats
    
    def optimize_for_inference(self):
        """Apply optimizations specifically for inference"""
        if torch.cuda.is_available():
            # Set optimal settings for inference
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient attention
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            logger.info("GPU optimized for inference performance")
    
    def preload_and_warmup(self):
        """Preload CUDA context and warm up GPU"""
        if torch.cuda.is_available():
            # Initialize CUDA context
            torch.cuda.init()
            
            # Warm up GPU with small operations
            dummy_tensor = torch.randn(100, 100, device='cuda')
            _ = torch.matmul(dummy_tensor, dummy_tensor)
            torch.cuda.synchronize()
            
            # Clear the dummy tensor
            del dummy_tensor
            torch.cuda.empty_cache()
            
            logger.info("GPU warmed up and ready for processing")

# Global optimizer instance
gpu_optimizer = GPUOptimizer()