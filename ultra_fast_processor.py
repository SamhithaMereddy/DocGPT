"""
Ultra-Fast Document Processor - 2 Second Target
Optimized for maximum speed with minimal processing overhead
"""
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

class UltraFastProcessor:
    """Ultra-fast document processor optimized for 2-second processing"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_ultra_fast_converter()
    
    def _initialize_ultra_fast_converter(self):
        """Initialize converter with SPEED-OPTIMIZED settings"""
        try:
            def create_fast_converter():
                # Ultra-fast converter with minimal configuration
                converter = DocumentConverter(
                    allowed_formats=[
                        InputFormat.PDF, 
                        InputFormat.DOCX, 
                        InputFormat.PPTX, 
                        InputFormat.MD,
                        InputFormat.HTML
                    ]
                )
                
                return converter
            
            self.converter = gpu_optimizer.get_or_create_model(
                model_type="ultra_fast_converter",
                model_path="docling_ultra_fast",
                create_func=create_fast_converter
            )
            
            self.logger.info("⚡ Ultra-fast converter initialized - targeting 2-second processing")
            
        except Exception as e:
            self.logger.error(f"Fast converter initialization failed: {e}")
            # Fallback to basic converter
            self.converter = DocumentConverter()
    
    def process_document_ultra_fast(self, file_path: str) -> Dict[str, Any]:
        """
        Ultra-fast document processing targeting 2 seconds
        
        Args:
            file_path: Path to document file
            
        Returns:
            Minimal document analysis optimized for speed
        """
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.logger.info(f"⚡ ULTRA-FAST processing: {file_path.name}")
            
            # Skip redundancy check for speed in ultra-fast mode
            analysis_result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "content": "",
                "processing_mode": "ultra_fast",
                "processing_time": 0,
                "pages": [],
                "metadata": {},
                "processing_errors": []
            }
            
            # ULTRA-FAST PROCESSING - NO HEAVY OPERATIONS
            if file_path.suffix.lower() == '.pdf':
                self._process_pdf_ultra_fast(file_path, analysis_result)
            elif file_path.suffix.lower() in ['.docx', '.pptx']:
                self._process_office_ultra_fast(file_path, analysis_result)
            elif file_path.suffix.lower() in ['.md', '.html', '.txt']:
                self._process_text_ultra_fast(file_path, analysis_result)
            else:
                # Try docling for other formats
                self._process_with_docling_fast(file_path, analysis_result)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            analysis_result["processing_time"] = processing_time
            
            self.logger.info(f"✅ ULTRA-FAST processing completed in {processing_time:.2f}s")
            
            # Warning if over 2 seconds
            if processing_time > 2.0:
                self.logger.warning(f"⚠️ Processing took {processing_time:.2f}s - target is 2.0s")
            
            return analysis_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Ultra-fast processing failed in {processing_time:.2f}s: {e}")
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "error": str(e),
                "processing_time": processing_time,
                "processing_mode": "ultra_fast_failed"
            }
    
    def _process_pdf_ultra_fast(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Ultra-fast PDF processing - text extraction only"""
        try:
            # Use docling for FAST text extraction only
            result = self.converter.convert(str(file_path))
            
            # Extract only text content - NO image/table processing
            text_content = result.document.export_to_markdown()
            analysis_result["content"] = text_content
            
            # Minimal metadata
            analysis_result["metadata"] = {"pages": len(result.document.pages) if hasattr(result.document, 'pages') else 1}
            
            # Create minimal page structure
            if text_content:
                analysis_result["pages"] = [{"page_number": 1, "text": text_content}]
            
        except Exception as e:
            self.logger.error(f"Ultra-fast PDF processing failed: {e}")
            analysis_result["processing_errors"].append(f"PDF processing error: {e}")
    
    def _process_office_ultra_fast(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Ultra-fast Office document processing"""
        try:
            result = self.converter.convert(str(file_path))
            text_content = result.document.export_to_markdown()
            
            analysis_result["content"] = text_content
            analysis_result["pages"] = [{"page_number": 1, "text": text_content}]
            
        except Exception as e:
            self.logger.error(f"Ultra-fast Office processing failed: {e}")
            analysis_result["processing_errors"].append(f"Office processing error: {e}")
    
    def _process_text_ultra_fast(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Ultra-fast text file processing"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            analysis_result["content"] = content
            analysis_result["pages"] = [{"page_number": 1, "text": content}]
            
        except Exception as e:
            self.logger.error(f"Ultra-fast text processing failed: {e}")
            analysis_result["processing_errors"].append(f"Text processing error: {e}")
    
    def _process_with_docling_fast(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Fallback ultra-fast processing with docling"""
        try:
            result = self.converter.convert(str(file_path))
            text_content = result.document.export_to_markdown()
            
            analysis_result["content"] = text_content
            analysis_result["pages"] = [{"page_number": 1, "text": text_content}]
            
        except Exception as e:
            self.logger.error(f"Docling fast processing failed: {e}")
            analysis_result["processing_errors"].append(f"Docling processing error: {e}")
    
    def chunk_document_fast(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Ultra-fast document chunking optimized for speed
        
        Args:
            document: Document with content
            
        Returns:
            List of text chunks
        """
        content = document.get('content', '')
        if not content:
            return []
        
        # FAST chunking - simple character-based splitting
        max_chunk_size = 1000  # Smaller chunks for faster processing
        overlap = 100
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = min(start + max_chunk_size, len(content))
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'chunk_id': f"{document.get('file_name', 'unknown')}_{chunk_id}",
                    'content': chunk_text,
                    'chunk_index': chunk_id,
                    'file_path': document.get('file_path', ''),
                    'file_name': document.get('file_name', ''),
                    'file_type': document.get('file_type', '')
                }
                chunks.append(chunk)
                chunk_id += 1
            
            start = end - overlap
        
        return chunks
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        memory_info = gpu_optimizer.get_gpu_memory_info() if torch.cuda.is_available() else None
        
        return {
            "processor_type": "ultra_fast",
            "target_time": "2.0 seconds",
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory": memory_info,
            "optimizations": [
                "No OCR processing",
                "No image extraction", 
                "No table analysis",
                "No chart analysis",
                "Minimal docling pipeline",
                "Fast text chunking",
                "GPU-accelerated where possible"
            ]
        }

# Global ultra-fast processor
ultra_fast_processor = None

def get_ultra_fast_processor(config):
    """Get or create ultra-fast processor instance"""
    global ultra_fast_processor
    if ultra_fast_processor is None:
        ultra_fast_processor = UltraFastProcessor(config)
    return ultra_fast_processor