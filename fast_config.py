"""
Ultra-Fast Processing Configuration
Optimized settings for 2-second document processing
"""
import os
from pathlib import Path

class UltraFastConfig:
    """Configuration optimized for ultra-fast document processing"""
    
    # SPEED-FIRST PROCESSING MODE
    ULTRA_FAST_MODE = True
    TARGET_PROCESSING_TIME = 2.0  # seconds
    
    # DISABLE HEAVY OPERATIONS FOR SPEED
    SKIP_OCR_PROCESSING = True          # Skip TrOCR for speed
    SKIP_IMAGE_EXTRACTION = True        # Skip image processing
    SKIP_TABLE_ANALYSIS = True          # Skip table extraction
    SKIP_CHART_ANALYSIS = True          # Skip chart analysis
    SKIP_HANDWRITING_DETECTION = True  # Skip handwriting recognition
    
    # DOCLING OPTIMIZATION
    DOCLING_MINIMAL_PIPELINE = True
    DOCLING_SKIP_IMAGES = True
    DOCLING_SKIP_TABLES = True
    DOCLING_TEXT_ONLY = True
    
    # CHUNKING OPTIMIZATION
    FAST_CHUNK_SIZE = 1000      # Smaller chunks for faster processing
    FAST_CHUNK_OVERLAP = 100    # Minimal overlap
    SIMPLE_CHUNKING = True      # Use simple character-based chunking
    
    # GPU OPTIMIZATION
    USE_GPU_ACCELERATION = True
    GPU_BATCH_SIZE_AGGRESSIVE = True
    GPU_MEMORY_AGGRESSIVE = True
    
    # I/O OPTIMIZATION
    MINIMAL_FILE_VALIDATION = True      # Skip heavy file validation
    FAST_JSON_SAVING = True             # Skip pretty-printing JSON
    SKIP_DETAILED_METADATA = True      # Save minimal metadata only
    
    # PROCESSING LIMITS
    MAX_FILE_SIZE_MB = 50               # Skip very large files
    MIN_CONTENT_LENGTH = 5              # Minimal content check
    MAX_PROCESSING_TIME_PER_FILE = 2.0  # Hard timeout per file
    
    # EMBEDDING OPTIMIZATION
    EMBEDDING_BATCH_SIZE_LARGE = 256    # Large batches for GPU
    EMBEDDING_SKIP_LONG_TEXTS = True    # Skip very long documents
    
    # LOGGING OPTIMIZATION
    REDUCE_LOGGING = True               # Minimize log output for speed
    SKIP_PROGRESS_BARS = True           # No progress bars for speed
    
    @classmethod
    def apply_ultra_fast_settings(cls):
        """Apply ultra-fast settings to environment"""
        
        # Set environment variables for speed
        os.environ['ULTRA_FAST_MODE'] = 'true'
        os.environ['DOCLING_SKIP_OCR'] = 'true'
        os.environ['DOCLING_SKIP_IMAGES'] = 'true'
        os.environ['DOCLING_MINIMAL_PIPELINE'] = 'true'
        
        # PyTorch optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        # Disable unnecessary warnings for speed
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        print("⚡ Ultra-fast processing settings applied")
        print("🎯 Target: 2-second document processing")
        print("🚀 GPU acceleration enabled")
        print("⚠️  Note: Some analysis features disabled for speed")
    
    @classmethod
    def get_speed_report(cls) -> str:
        """Generate speed optimization report"""
        report = "\n⚡ ULTRA-FAST PROCESSING CONFIGURATION\n"
        report += "=" * 50 + "\n"
        report += "🎯 Target Processing Time: 2.0 seconds\n\n"
        
        report += "DISABLED FOR SPEED:\n"
        report += "❌ OCR Processing (TrOCR)\n"
        report += "❌ Image Extraction & Analysis\n"
        report += "❌ Table Structure Analysis\n"
        report += "❌ Chart & Graph Analysis\n"
        report += "❌ Handwriting Recognition\n"
        report += "❌ Complex Document Validation\n\n"
        
        report += "OPTIMIZATIONS ENABLED:\n"
        report += "✅ GPU-Accelerated Text Extraction\n"
        report += "✅ Minimal Docling Pipeline\n"
        report += "✅ Fast Text Chunking\n"
        report += "✅ Aggressive GPU Batching\n"
        report += "✅ Minimal I/O Operations\n"
        report += "✅ Streamlined JSON Storage\n\n"
        
        report += "PROCESSING PIPELINE:\n"
        report += "1. Quick file validation\n"
        report += "2. GPU-accelerated text extraction\n"
        report += "3. Fast text chunking\n"
        report += "4. Minimal metadata generation\n"
        report += "5. Fast JSON storage\n\n"
        
        report += "TRADE-OFFS:\n"
        report += "⚡ 10x faster processing\n"
        report += "📄 Text-only analysis\n"
        report += "🎯 Optimized for search & retrieval\n"
        report += "⚠️  Reduced analysis depth\n"
        
        report += "=" * 50
        
        return report

# Apply ultra-fast settings on import
UltraFastConfig.apply_ultra_fast_settings()

# Export configuration
ultra_fast_config = UltraFastConfig()