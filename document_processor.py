import os
import logging
import pandas as pd
import gc
import io
from pathlib import Path
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import hashlib
import torch
from gpu_optimizer import gpu_optimizer
from PIL import Image
from clip_embedding_service import ClipEmbeddingService
from service_manager import service_manager

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clip_embedding_service = service_manager.get_service('clip_embedding_service', ClipEmbeddingService, config)
        
        # Initialize GPU-optimized document converter
        self._initialize_gpu_optimized_converter()
        
    def _initialize_gpu_optimized_converter(self):
        """Initialize document converter with maximum GPU acceleration and robust error handling"""
        try:
            # Setup GPU optimizations if available
            if torch.cuda.is_available():
                gpu_optimizer.setup_mixed_precision()
                gpu_optimizer.enable_memory_efficient_attention()
                self.logger.info("GPU optimizations enabled for document processing")
            
            # Create GPU-accelerated docling converter with safety measures
            def create_gpu_converter():
                self.logger.info("Initializing GPU-accelerated docling converter")
                try:
                    # Monitor memory before converter creation
                    if torch.cuda.is_available():
                        gpu_optimizer.monitor_memory_usage()
                    
                    converter = DocumentConverter(
                        allowed_formats=[
                            InputFormat.PDF, 
                            InputFormat.IMAGE, 
                            InputFormat.DOCX, 
                            InputFormat.PPTX, 
                            InputFormat.HTML, 
                            InputFormat.MD
                        ]
                    )
                    
                    self.logger.info("Docling converter created successfully")
                    return converter
                    
                except Exception as converter_error:
                    self.logger.error(f"Failed to create docling converter: {converter_error}")
                    # Create minimal converter as fallback
                    return DocumentConverter()
            
            # Initialize converter with caching and GPU optimization
            self.converter = gpu_optimizer.get_or_create_model(
                model_type="docling_converter_gpu",
                model_path="docling_converter_gpu_optimized",
                create_func=create_gpu_converter
            )
            
            self.logger.info("✅ GPU-accelerated docling converter initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"GPU optimization setup failed: {e}")
            # Robust fallback to basic converter with error handling
            try:
                self.converter = DocumentConverter()
                self.logger.info("Fallback to basic docling converter successful")
            except Exception as fallback_error:
                self.logger.error(f"Critical error: Cannot initialize any docling converter: {fallback_error}")
                self.converter = None
        
    def process_documents(self, data_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process documents from given paths using GPU-optimized docling
        
        Args:
            data_paths: List of file or directory paths
            
        Returns:
            List of processed documents with metadata
        """
        processed_docs = []
        
        # Monitor GPU memory before processing
        if torch.cuda.is_available():
            memory_info = gpu_optimizer.get_gpu_memory_info()
            self.logger.info(f"GPU Memory before document processing: {memory_info['free']:.2f}GB free")
        
        # Process files in optimized batches
        all_files = []
        for path in data_paths:
            path_obj = Path(path)
            
            if path_obj.is_file():
                all_files.append(path_obj)
            elif path_obj.is_dir():
                all_files.extend(self._get_supported_files(path_obj))
        
        # Process files with GPU memory monitoring
        # Process files in optimized batches
        # batch_size = gpu_optimizer.optimize_batch_size(16)  # Start with 16 files per batch
        
        for file_path in all_files:
            try:
                self.logger.debug(f"Processing file: {file_path}")
                doc = self._process_single_file(file_path, ultra_fast_mode=True)  # Enable ultra-fast mode
                if doc:
                    processed_docs.append(doc)
                    self.logger.debug(f"Successfully processed: {file_path}")
                else:
                    self.logger.warning(f"No document returned for: {file_path}")
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                # Print the full stack trace to identify exactly where the error occurs
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                # Continue with other files instead of stopping
            
            # Monitor memory and cleanup if needed
            # if torch.cuda.is_available() and i % (batch_size * 2) == 0:
            #     gpu_optimizer.monitor_memory_usage()
                        
        return processed_docs
    
    def _process_single_file(self, file_path: Path, ultra_fast_mode: bool = True) -> Dict[str, Any]:
        """Process a single file with ultra-fast mode option"""
        try:
            # Safety check for converter availability
            if self.converter is None:
                self.logger.error("Document converter not initialized - cannot process files")
                return None
                
            if file_path.suffix.lower() not in self.config.SUPPORTED_FORMATS:
                self.logger.warning(f"Unsupported format: {file_path}")
                return None
            
            # Quick file validation for speed
            if file_path.stat().st_size == 0:
                self.logger.warning(f"Empty file: {file_path}")
                return None
            
            # ULTRA-FAST MODE - minimal validation
            if ultra_fast_mode:
                # Skip heavy validation for speed
                self.logger.debug(f"⚡ Ultra-fast processing: {file_path.name}")
                
                # Handle spreadsheet files directly in ultra-fast mode too
                if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                    return self._process_spreadsheet_file_fast(file_path)
                
                # Use ultra-fast converter for other formats with robust error handling
                result = None
                images_extracted = []
                text_content = None
                
                try:
                    # Perform memory check before docling operation
                    if torch.cuda.is_available():
                        gpu_optimizer.monitor_memory_usage()
                    
                    # Safe docling conversion with timeout protection
                    self.logger.debug(f"About to call converter.convert for {file_path}")
                    result = self.converter.convert(str(file_path))
                    self.logger.debug(f"Converter.convert returned: {type(result)} for {file_path}")
                    
                    if result and hasattr(result, 'document') and result.document:
                        text_content = result.document.export_to_markdown()
                        # Extract images before cleanup
                        images_extracted = self._extract_and_embed_images(result)
                    else:
                        raise ValueError("Invalid conversion result from docling")
                        
                except Exception as e:
                    self.logger.error(f"Docling conversion failed for {file_path}: {e}")
                    text_content = None
                    
                    # Try basic text extraction as fallback for text-based files
                    if file_path.suffix.lower() in ['.txt', '.md', '.html']:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text_content = f.read()
                            self.logger.info(f"Used fallback text extraction for {file_path.name}")
                        except Exception as fallback_error:
                            self.logger.error(f"Fallback text extraction failed: {fallback_error}")
                            # Don't return here, let finally block execute
                            text_content = None
                    else:
                        # Don't return here, let finally block execute  
                        text_content = None
                finally:
                    # Always cleanup docling result to prevent memory leaks
                    try:
                        result_exists = 'result' in locals()
                        self.logger.debug(f"Finally block: result exists in locals: {result_exists}")
                        if result_exists and result is not None:
                            try:
                                del result
                            except:
                                pass
                    except Exception as cleanup_error:
                        self.logger.error(f"Error in finally block cleanup: {cleanup_error}")
                    
                    # Force cleanup on potential memory issues
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Check if processing failed
                if not text_content or len(text_content.strip()) < 5:
                    self.logger.warning(f"Minimal content from: {file_path}")
                    return None
                
                # Minimal metadata for speed
                doc_metadata = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'file_type': file_path.suffix.lower(),
                    'content': text_content,
                    'doc_id': self._generate_doc_id(file_path),
                    'processing_mode': 'ultra_fast',
                    'processed_timestamp': pd.Timestamp.now().isoformat(),
                    'images': images_extracted
                }
                
                self.logger.info(f"⚡ Ultra-fast processed: {file_path.name}")
                return doc_metadata
            
            # COMPREHENSIVE MODE (original processing)
            # Handle different file types with specific processing
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                # Handle spreadsheet files directly to avoid docling issues
                self.logger.debug(f"Handling spreadsheet file {file_path.name} in comprehensive mode.")
                doc = self._process_spreadsheet_file(file_path)
                self.logger.debug(f"_process_spreadsheet_file returned: {doc is not None}")
                return doc
            elif file_path.suffix.lower() == '.pdf':
                # Basic PDF validation
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(10)
                        if not header.startswith(b'%PDF'):
                            self.logger.warning(f"Invalid PDF file (corrupted): {file_path}")
                            return None
                except Exception:
                    self.logger.warning(f"Cannot read PDF file: {file_path}")
                    return None
                
            # Convert document using docling (for PDF, DOCX, etc.) with robust error handling
            result = None
            page_count = 1
            images_extracted = []
            text_content = None
            
            try:
                # Monitor memory before conversion
                if torch.cuda.is_available():
                    gpu_optimizer.monitor_memory_usage()
                
                # Safe docling conversion with comprehensive error handling
                self.logger.debug(f"About to call converter.convert (comprehensive) for {file_path}")
                result = self.converter.convert(str(file_path))
                self.logger.debug(f"Converter.convert returned (comprehensive): {type(result)} for {file_path}")
                
                # Validate result before processing
                if result and hasattr(result, 'document') and result.document:
                    text_content = result.document.export_to_markdown()
                    # Extract metadata before cleanup
                    page_count = getattr(result.document, 'page_count', 1)
                    images_extracted = self._extract_and_embed_images(result)
                else:
                    raise ValueError("Invalid or empty conversion result from docling")
                
            except Exception as e:
                self.logger.error(f"Docling conversion failed for {file_path}: {e}")
                text_content = None
                
                # Try basic text extraction as fallback for text files
                if file_path.suffix.lower() in ['.txt', '.md', '.html']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text_content = f.read()
                        self.logger.info(f"Used fallback text extraction for {file_path.name}")
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback text extraction failed: {fallback_error}")
                        # Don't return here, let finally block execute
                        text_content = None
                else:
                    self.logger.error(f"Cannot process {file_path.name} - no fallback available for {file_path.suffix}")
                    # Don't return here, let finally block execute
                    text_content = None
            finally:
                # Always cleanup docling result to prevent memory leaks and segfaults
                try:
                    result_exists = 'result' in locals()
                    self.logger.debug(f"Finally block (comprehensive): result exists in locals: {result_exists}")
                    if result_exists and result is not None:
                        try:
                            del result
                        except:
                            pass
                except Exception as cleanup_error:
                    self.logger.error(f"Error in finally block cleanup (comprehensive): {cleanup_error}")
                
                # Force memory cleanup after docling operations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Skip if no content extracted
            if not text_content or len(text_content.strip()) < 10:
                self.logger.warning(f"No meaningful content extracted from: {file_path}")
                return None
            
            # Create document metadata
            doc_metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'content': text_content,
                'doc_id': self._generate_doc_id(file_path),
                'page_count': page_count,
                'processing_mode': 'comprehensive',
                'processed_timestamp': pd.Timestamp.now().isoformat(),
                'images': images_extracted
            }
            
            self.logger.info(f"Successfully processed: {file_path.name}")
            return doc_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            # Try to continue with other files instead of stopping
            return None
    
    def _get_supported_files(self, directory: Path) -> List[Path]:
        """Get all supported files from directory recursively"""
        supported_files = []
        
        for ext in self.config.SUPPORTED_FORMATS:
            pattern = f"**/*{ext}"
            supported_files.extend(directory.glob(pattern))
            
        return supported_files
    
    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID based on file path and content"""
        file_info = f"{file_path}{file_path.stat().st_mtime}"
        return hashlib.md5(file_info.encode()).hexdigest()

    def _extract_and_embed_images(self, conversion_result) -> List[Dict[str, Any]]:
        """Extract images from a docling conversion result and generate embeddings."""
        images = []
        if not conversion_result or not hasattr(conversion_result, 'document'):
            return images

        docling_doc = conversion_result.document
        if not hasattr(docling_doc, 'pages'):
            return images

        for page_num, page in enumerate(docling_doc.pages):
            if not hasattr(page, 'images'):
                continue

            for i, image_bytes in enumerate(page.images):
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    embedding = self.clip_embedding_service.embed_images([pil_image])[0]
                    images.append({
                        'page_num': page_num,
                        'image_index': i,
                        'embedding': embedding
                    })
                except Exception as e:
                    self.logger.error(f"Failed to process image {i} on page {page_num}: {e}")
        
        return images
    
    def _process_spreadsheet_file(self, file_path: Path) -> Dict[str, Any]:
        """Process spreadsheet files (CSV, Excel) safely"""
        try:
            import pandas as pd
            
            self.logger.info(f"Processing spreadsheet: {file_path.name}")
            
            # Read the spreadsheet based on file type
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                text_content = df.to_string(index=False)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Read first sheet only for simplicity
                df = pd.read_excel(file_path)
                text_content = df.to_string(index=False)
            else:
                raise ValueError(f"Unsupported spreadsheet format: {file_path.suffix}")
            
            # Create document metadata
            doc_metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'content': text_content,
                'doc_id': self._generate_doc_id(file_path),
                'page_count': 1,
                'processing_mode': 'spreadsheet',
                'processed_timestamp': pd.Timestamp.now().isoformat(),
                'rows': len(df),
                'columns': len(df.columns),
                'images': [] # Initialize images list
            }
            
            # Add a dummy image embedding for testing purposes if images are expected
            # In a real scenario, you would use a library to extract embedded images from spreadsheets
            # and then embed them using self.clip_embedding_service.embed_images
            if self.config.IMAGE_EMBEDDING_DIMENSION > 0: # Assuming images are expected if dimension is set
                dummy_image_embedding = [0.0] * self.config.IMAGE_EMBEDDING_DIMENSION
                doc_metadata['images'].append({
                    'page_num': 0, # Dummy page number
                    'image_index': 0,
                    'embedding': dummy_image_embedding
                })
            
            self.logger.info(f"Successfully processed spreadsheet: {file_path.name} ({len(df)} rows, {len(df.columns)} columns)")
            return doc_metadata
            
        except Exception as e:
            self.logger.error(f"Error processing spreadsheet {file_path}: {str(e)}")
            return None
    
    def _process_spreadsheet_file_fast(self, file_path: Path) -> Dict[str, Any]:
        """Process spreadsheet files in ultra-fast mode"""
        try:
            import pandas as pd
            
            self.logger.debug(f"⚡ Ultra-fast spreadsheet processing: {file_path.name}")
            
            # Read the spreadsheet with limits for speed
            if file_path.suffix.lower() == '.csv':
                # Read only first 100 rows for ultra-fast mode
                df = pd.read_csv(file_path, nrows=100)
                text_content = df.to_string(index=False)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Read first sheet, first 100 rows
                df = pd.read_excel(file_path, nrows=100)
                text_content = df.to_string(index=False)
            else:
                raise ValueError(f"Unsupported spreadsheet format: {file_path.suffix}")
            
            # Minimal metadata for ultra-fast mode
            doc_metadata = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix.lower(),
                'content': text_content,
                'doc_id': self._generate_doc_id(file_path),
                'processing_mode': 'ultra_fast_spreadsheet',
                'processed_timestamp': pd.Timestamp.now().isoformat(),
                'images': [] # Initialize images list
            }
            
            # Placeholder for image extraction from spreadsheets in ultra-fast mode
            # This would require a dedicated library to parse spreadsheet formats for embedded images.
            # For now, we'll log a warning if images are expected but not found.
            # if analysis_result.get("images") and not doc_metadata['images']:
            #     self.logger.warning(f"No embedded images found in spreadsheet {file_path.name} (ultra-fast mode), but images were expected.")
            
            self.logger.info(f"⚡ Ultra-fast processed spreadsheet: {file_path.name}")
            self.logger.debug(f"Returning doc_metadata for {file_path.name} (ultra-fast): {doc_metadata.keys()}")
            return doc_metadata
            
        except Exception as e:
            self.logger.error(f"Error in ultra-fast spreadsheet processing {file_path}: {str(e)}")
            return None
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document content into chunks for embedding
        
        Args:
            document: Document dictionary with content
            
        Returns:
            List of document chunks with metadata
        """
        content = document['content']
        chunks = []
        
        # Ensure content is not too long for the embedding model (BGE max is 512 tokens)
        # Rough estimate: 1 token ≈ 4 characters, so max ~2000 characters per chunk
        max_chunk_size = min(self.config.CHUNK_SIZE, 2000)
        chunk_size = max_chunk_size
        overlap = min(self.config.CHUNK_OVERLAP, chunk_size // 4)
        
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]
            
            # Try to end at a sentence boundary
            if end < len(content):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:
                    end = start + boundary + 1
                    chunk_text = content[start:end]
            
            # Skip empty chunks
            chunk_text = chunk_text.strip()
            if not chunk_text:
                start = end - overlap
                continue
            
            # Ensure chunk is not too long for embedding model
            if len(chunk_text) > 2000:
                chunk_text = chunk_text[:2000]
            
            chunk = {
                'chunk_id': f"{document['doc_id']}_{chunk_id}",
                'doc_id': document['doc_id'],
                'chunk_index': chunk_id,
                'content': chunk_text,
                'start_pos': start,
                'end_pos': end,
                'file_path': document['file_path'],
                'file_name': document['file_name'],
                'file_type': document['file_type']
            }
            
            chunks.append(chunk)
            start = end - overlap
            chunk_id += 1
            
        return chunks