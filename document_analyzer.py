"""
Enhanced Document Analyzer with table extraction, handwriting recognition, and source tracking
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import torch
from document_tracker import DocumentTracker

document_tracker = DocumentTracker()
from chart_analyzer import chart_analyzer

# Core document processing
import fitz  # PyMuPDF for PDF processing
import cv2
import pytesseract

# Table extraction
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("Camelot not available - table extraction from PDFs limited")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    print("Tabula not available - alternative table extraction will be used")

# OCR and handwriting recognition
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("TrOCR not available - using pytesseract only")

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """Enhanced document analyzer with advanced features"""
    
    def __init__(self, config=None):
        self.config = config
        self.uploads_dir = Path("uploads")
        self.processed_dir = Path("processed_files")
        self.uploads_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize GPU-accelerated TrOCR with maximum optimization
        if TROCR_AVAILABLE:
            try:
                from gpu_optimizer import gpu_optimizer
                
                # Setup GPU optimizations first
                if torch.cuda.is_available():
                    gpu_optimizer.setup_mixed_precision()
                    gpu_optimizer.enable_memory_efficient_attention()
                
                def create_trocr_components():
                    # Use fast processor to avoid slow warning
                    processor = TrOCRProcessor.from_pretrained(
                        'microsoft/trocr-base-printed',
                        use_fast=True
                    )
                    
                    model = VisionEncoderDecoderModel.from_pretrained(
                        'microsoft/trocr-base-printed',
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    # Aggressive GPU optimization
                    if torch.cuda.is_available():
                        model = model.to('cuda').half()
                        
                        # Enable memory efficient optimizations
                        if hasattr(model, 'encoder'):
                            if hasattr(model.encoder, 'gradient_checkpointing_enable'):
                                model.encoder.gradient_checkpointing_enable()
                        
                        # Compile model for maximum speed (PyTorch 2.0+)
                        if hasattr(torch, 'compile'):
                            try:
                                model = torch.compile(model, mode='max-autotune')
                                logger.info("TrOCR model compiled for maximum GPU performance")
                            except Exception as compile_error:
                                logger.debug(f"Model compilation not available: {compile_error}")
                    
                    return processor, model
                
                # Use model caching for faster loading
                cached_components = gpu_optimizer.get_or_create_model(
                    model_type="trocr_components",
                    model_path="microsoft/trocr-base-printed",
                    create_func=create_trocr_components
                )
                
                self.trocr_processor, self.trocr_model = cached_components
                
                if torch.cuda.is_available():
                    memory_info = gpu_optimizer.get_gpu_memory_info()
                    logger.info(f"✅ TrOCR initialized on GPU - {memory_info['allocated']:.2f}GB VRAM used")
                else:
                    logger.info("✅ TrOCR initialized on CPU")
                    
            except Exception as e:
                logger.warning(f"TrOCR initialization failed: {e}")
                self.trocr_processor = None
                self.trocr_model = None
        else:
            self.trocr_processor = None
            self.trocr_model = None
    
    def save_uploaded_file(self, uploaded_file, filename: str = None) -> str:
        """Save uploaded file permanently and return file path"""
        try:
            if filename is None:
                filename = uploaded_file.name
            
            # Ensure unique filename
            file_path = self.uploads_dir / filename
            counter = 1
            while file_path.exists():
                name, ext = os.path.splitext(filename)
                file_path = self.uploads_dir / f"{name}_{counter}{ext}"
                counter += 1
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def analyze_document(self, file_path: str, ultra_fast_mode: bool = True, optimized_mode: bool = True) -> Dict[str, Any]:
        """Document analysis with ultra-fast mode for 2-second processing"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if document already processed
            is_processed, existing_doc_info = document_tracker.is_document_processed(str(file_path))
            
            if is_processed:
                logger.info(f"Document already processed: {file_path.name}")
                return document_tracker.handle_duplicate_document(str(file_path), existing_doc_info)
            
            # ULTRA-FAST MODE for 2-second processing
            if ultra_fast_mode:
                from ultra_fast_processor import get_ultra_fast_processor
                
                logger.info(f"⚡ ULTRA-FAST analysis: {file_path.name}")
                ultra_processor = get_ultra_fast_processor(self.config)
                
                # Use ultra-fast processor
                analysis_result = ultra_processor.process_document_ultra_fast(str(file_path))
                analysis_result["is_new_document"] = True
                analysis_result["processing_mode"] = "ultra_fast"
                
                # Minimal result saving for speed
                self._save_analysis_results_fast(analysis_result)
                
                # Register document in tracking system
                analysis_result = document_tracker.register_document(str(file_path), analysis_result)
                
                return analysis_result
            
            # OPTIMIZED COMPREHENSIVE MODE
            mode_name = "optimized_comprehensive" if optimized_mode else "comprehensive"
            logger.info(f"{mode_name} analysis: {file_path.name}")
            
            analysis_result = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower(),
                "pages": [],
                "tables": [],
                "images": [],
                "handwriting_text": [],
                "metadata": {},
                "processing_errors": [],
                "is_new_document": True,
                "processing_mode": mode_name,
                "optimizations_enabled": optimized_mode
            }
            
            # Process based on file type
            if file_path.suffix.lower() == '.pdf':
                self._analyze_pdf(file_path, analysis_result, optimized_mode)
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                self._analyze_image(file_path, analysis_result)
            elif file_path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
                self._analyze_spreadsheet(file_path, analysis_result)
            else:
                # Try to process as text
                self._analyze_text_file(file_path, analysis_result)
            
            # Save analysis results
            self._save_analysis_results(analysis_result)
            
            # Register document in tracking system
            analysis_result = document_tracker.register_document(str(file_path), analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "error": str(e),
                "processing_errors": [str(e)]
            }
    
    def _analyze_pdf(self, file_path: Path, analysis_result: Dict[str, Any], optimized_mode: bool = True):
        """GPU-accelerated PDF document analysis"""
        try:
            from gpu_optimizer import gpu_optimizer
            from gpu_image_processor import gpu_image_processor
            
            # Monitor GPU memory before processing
            if torch.cuda.is_available():
                initial_memory = gpu_optimizer.get_gpu_memory_info()
                logger.info(f"Starting PDF analysis - GPU Memory: {initial_memory['free']:.2f}GB free")
            
            doc = fitz.open(file_path)
            analysis_result["metadata"] = doc.metadata
            
            # Process pages with optimizations
            if optimized_mode:
                # Optimized mode: Process fewer pages and limit image processing
                max_pages = min(len(doc), 3)  # Limit to first 3 pages for speed
                batch_size = 1  # Process one page at a time
                logger.info(f"Optimized mode: Processing first {max_pages} pages only")
            else:
                # Full mode: Process all pages
                max_pages = len(doc)
                batch_size = gpu_optimizer.optimize_batch_size(4)
                logger.info(f"Full mode: Processing all {max_pages} pages")
            
            for batch_start in range(0, max_pages, batch_size):
                batch_end = min(batch_start + batch_size, max_pages)
                logger.info(f"Processing PDF pages {batch_start + 1}-{batch_end}")
                
                for page_num in range(batch_start, batch_end):
                    page = doc.load_page(page_num)
                    page_analysis = {
                        "page_number": page_num + 1,
                        "text": "",
                        "images": [],
                        "tables": [],
                        "handwriting": []
                    }
                    
                    # Extract text
                    page_analysis["text"] = page.get_text()
                    
                    # GPU-accelerated image extraction and processing
                    image_list = page.get_images()
                    if image_list:
                        # Optimized mode: Limit number of images processed per page
                        if optimized_mode:
                            image_list = image_list[:1]  # Process only 1 image per page for speed
                        
                        # Batch process images for GPU efficiency
                        pil_images = []
                        image_paths = []
                        
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                
                                # Save image
                                image_filename = f"{file_path.stem}_page{page_num+1}_img{img_index+1}.{image_ext}"
                                image_path = self.processed_dir / image_filename
                                
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_bytes)
                                
                                # Load image for GPU processing with CMYK handling
                                from PIL import Image
                                try:
                                    pil_img = Image.open(image_path)
                                    # Convert CMYK to RGB to avoid processing errors
                                    if pil_img.mode == 'CMYK':
                                        pil_img = pil_img.convert('RGB')
                                    elif pil_img.mode not in ['RGB', 'L', 'P']:
                                        pil_img = pil_img.convert('RGB')
                                    pil_images.append(pil_img)
                                    image_paths.append(image_path)
                                except Exception as img_error:
                                    self.logger.warning(f"Could not process image {image_path}: {img_error}")
                                    # Continue with other images
                                
                            except Exception as e:
                                analysis_result["processing_errors"].append(f"Error extracting image on page {page_num+1}: {e}")
                        
                        # GPU batch processing of images
                        if pil_images:
                            try:
                                # Process images in GPU batches for better performance
                                for i, (pil_img, img_path) in enumerate(zip(pil_images, image_paths)):
                                    # GPU-accelerated image analysis with optimizations
                                    if optimized_mode:
                                        # Optimized: Skip TrOCR and heavy chart analysis, only basic OCR
                                        image_analysis = self._analyze_image_content_optimized(img_path)
                                        # Skip chart analysis for speed
                                        image_analysis.update({
                                            "chart_analysis": {"chart_type": None, "insights": []},
                                            "is_chart": False,
                                            "visual_insights": []
                                        })
                                    else:
                                        # Full analysis with TrOCR
                                        image_analysis = self._analyze_image_content(img_path)
                                        # Advanced chart analysis
                                        chart_analysis = chart_analyzer.analyze_image_content(str(img_path))
                                        image_analysis.update({
                                            "chart_analysis": chart_analysis,
                                            "is_chart": chart_analysis.get("chart_type") is not None,
                                            "visual_insights": chart_analysis.get("insights", [])
                                        })
                                    
                                    page_analysis["images"].append({
                                        "image_path": str(img_path),
                                        "image_index": i,
                                        "analysis": image_analysis
                                    })
                                    
                                    analysis_result["images"].append({
                                        "page": page_num + 1,
                                        "path": str(img_path),
                                        "analysis": image_analysis
                                    })
                                    
                            except Exception as e:
                                analysis_result["processing_errors"].append(f"Error in GPU image processing on page {page_num+1}: {e}")
                    
                    # Extract tables from PDF
                    self._extract_pdf_tables(file_path, page_num + 1, analysis_result)
                    
                    analysis_result["pages"].append(page_analysis)
                
                # Monitor GPU memory after batch processing
                if torch.cuda.is_available() and batch_start % (batch_size * 2) == 0:
                    gpu_optimizer.monitor_memory_usage()
            
            doc.close()
            
        except Exception as e:
            analysis_result["processing_errors"].append(f"PDF analysis error: {e}")
    
    def _analyze_image(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Analyze image file"""
        try:
            image_analysis = self._analyze_image_content(file_path)
            
            analysis_result["pages"] = [{
                "page_number": 1,
                "text": image_analysis.get("ocr_text", ""),
                "handwriting": image_analysis.get("handwriting_text", ""),
                "confidence": image_analysis.get("confidence", 0)
            }]
            
            if image_analysis.get("handwriting_text"):
                analysis_result["handwriting_text"].append({
                    "page": 1,
                    "text": image_analysis["handwriting_text"],
                    "confidence": image_analysis.get("confidence", 0)
                })
            
        except Exception as e:
            analysis_result["processing_errors"].append(f"Image analysis error: {e}")
    
    def _analyze_image_content(self, image_path: Path) -> Dict[str, Any]:
        """Analyze image content for text and handwriting"""
        try:
            # Load image with CMYK handling
            image = Image.open(image_path)
            # Convert CMYK and other problematic modes to RGB
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            elif image.mode not in ['RGB', 'L', 'P']:
                image = image.convert('RGB')
            
            result = {
                "ocr_text": "",
                "handwriting_text": "",
                "confidence": 0,
                "method_used": "pytesseract"
            }
            
            # Regular OCR with pytesseract
            try:
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                ocr_text = pytesseract.image_to_string(image)
                result["ocr_text"] = ocr_text.strip()
                
                # Calculate average confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                if confidences:
                    result["confidence"] = sum(confidences) / len(confidences)
                
            except Exception as e:
                logger.warning(f"Pytesseract OCR failed: {e}")
            
            # GPU-accelerated handwriting recognition with TrOCR
            if self.trocr_processor and self.trocr_model:
                try:
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # GPU-accelerated preprocessing
                    with torch.no_grad():
                        pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
                        
                        # Move pixel_values to the same device as the model
                        if torch.cuda.is_available() and next(self.trocr_model.parameters()).is_cuda:
                            pixel_values = pixel_values.to('cuda')
                        
                        # Use mixed precision for faster inference
                        if torch.cuda.is_available():
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                generated_ids = self.trocr_model.generate(
                                    pixel_values,
                                    max_length=384,  # Reasonable max length
                                    num_beams=1,     # Faster single beam
                                    do_sample=False, # Deterministic for speed
                                    pad_token_id=self.trocr_processor.tokenizer.pad_token_id,
                                    use_cache=True   # Enable KV cache
                                )
                        else:
                            generated_ids = self.trocr_model.generate(
                                pixel_values,
                                max_length=384,
                                num_beams=1,
                                do_sample=False,
                                pad_token_id=self.trocr_processor.tokenizer.pad_token_id,
                                use_cache=True
                            )
                    
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    result["handwriting_text"] = generated_text.strip()
                    result["method_used"] = "gpu_trocr+pytesseract"
                    
                except Exception as e:
                    logger.warning(f"GPU TrOCR failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image content: {e}")
            return {"error": str(e)}
    
    def _analyze_image_content_optimized(self, image_path: Path) -> Dict[str, Any]:
        """Optimized image analysis - OCR only, skip TrOCR for speed"""
        try:
            # Load image with CMYK handling
            image = Image.open(image_path)
            # Convert CMYK and other problematic modes to RGB
            if image.mode == 'CMYK':
                image = image.convert('RGB')
            elif image.mode not in ['RGB', 'L', 'P']:
                image = image.convert('RGB')
            
            result = {
                "ocr_text": "",
                "handwriting_text": "",  # Empty in optimized mode
                "confidence": 0,
                "method_used": "pytesseract_optimized"
            }
            
            # Fast OCR with pytesseract only
            try:
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                ocr_text = pytesseract.image_to_string(image, config='--psm 6')  # Fast PSM mode
                result["ocr_text"] = ocr_text.strip()
                
                # Calculate average confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                if confidences:
                    result["confidence"] = sum(confidences) / len(confidences)
                
            except Exception as e:
                logger.warning(f"Optimized OCR failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized image analysis: {e}")
            return {"error": str(e)}
    
    def _extract_pdf_tables(self, file_path: Path, page_num: int, analysis_result: Dict[str, Any]):
        """Extract tables from PDF using multiple methods"""
        try:
            # Try Camelot first (better for complex tables)
            if CAMELOT_AVAILABLE:
                try:
                    tables = camelot.read_pdf(str(file_path), pages=str(page_num))
                    
                    for i, table in enumerate(tables):
                        if table.df is not None and not table.df.empty:
                            table_data = {
                                "page": page_num,
                                "table_index": i,
                                "method": "camelot",
                                "accuracy": table.accuracy,
                                "data": table.df.to_dict('records'),
                                "csv_data": table.df.to_csv(index=False)
                            }
                            
                            # Save table as CSV
                            csv_filename = f"{file_path.stem}_page{page_num}_table{i+1}.csv"
                            csv_path = self.processed_dir / csv_filename
                            table.df.to_csv(csv_path, index=False)
                            table_data["csv_path"] = str(csv_path)
                            
                            analysis_result["tables"].append(table_data)
                
                except Exception as e:
                    logger.warning(f"Camelot table extraction failed for page {page_num}: {e}")
            
            # Try Tabula as fallback
            if TABULA_AVAILABLE and not analysis_result["tables"]:
                try:
                    dfs = tabula.read_pdf(str(file_path), pages=page_num, multiple_tables=True)
                    
                    for i, df in enumerate(dfs):
                        if not df.empty:
                            table_data = {
                                "page": page_num,
                                "table_index": i,
                                "method": "tabula",
                                "data": df.to_dict('records'),
                                "csv_data": df.to_csv(index=False)
                            }
                            
                            # Save table as CSV
                            csv_filename = f"{file_path.stem}_page{page_num}_table{i+1}_tabula.csv"
                            csv_path = self.processed_dir / csv_filename
                            df.to_csv(csv_path, index=False)
                            table_data["csv_path"] = str(csv_path)
                            
                            analysis_result["tables"].append(table_data)
                
                except Exception as e:
                    logger.warning(f"Tabula table extraction failed for page {page_num}: {e}")
        
        except Exception as e:
            analysis_result["processing_errors"].append(f"Table extraction error on page {page_num}: {e}")
    
    def _analyze_spreadsheet(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Analyze spreadsheet files"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                sheets = {"Sheet1": df}
            else:
                sheets = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, df in sheets.items():
                table_data = {
                    "sheet_name": sheet_name,
                    "method": "pandas",
                    "shape": df.shape,
                    "data": df.to_dict('records'),
                    "csv_data": df.to_csv(index=False),
                    "columns": df.columns.tolist()
                }
                
                analysis_result["tables"].append(table_data)
                
                # Also add as page content
                analysis_result["pages"].append({
                    "page_number": len(analysis_result["pages"]) + 1,
                    "sheet_name": sheet_name,
                    "text": df.to_string(),
                    "table_data": table_data
                })
        
        except Exception as e:
            analysis_result["processing_errors"].append(f"Spreadsheet analysis error: {e}")
    
    def _analyze_text_file(self, file_path: Path, analysis_result: Dict[str, Any]):
        """Analyze plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis_result["pages"] = [{
                "page_number": 1,
                "text": content
            }]
            
        except Exception as e:
            analysis_result["processing_errors"].append(f"Text file analysis error: {e}")
    
    def _save_analysis_results(self, analysis_result: Dict[str, Any]):
        """Save analysis results to JSON file"""
        try:
            file_path = Path(analysis_result["file_path"])
            results_filename = f"{file_path.stem}_analysis.json"
            results_path = self.processed_dir / results_filename
            
            # Make analysis_result serializable
            serializable_result = self._make_serializable(analysis_result)
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            analysis_result["analysis_file"] = str(results_path)
            logger.info(f"Analysis results saved: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def _save_analysis_results_fast(self, analysis_result: Dict[str, Any]):
        """Fast analysis results saving - minimal I/O for speed"""
        try:
            file_path = Path(analysis_result["file_path"])
            results_filename = f"{file_path.stem}_analysis_fast.json"
            results_path = self.processed_dir / results_filename
            
            # Save only essential data for speed
            minimal_result = {
                "file_path": analysis_result["file_path"],
                "file_name": analysis_result["file_name"],
                "file_type": analysis_result["file_type"],
                "content": analysis_result.get("content", ""),
                "processing_time": analysis_result.get("processing_time", 0),
                "processing_mode": "ultra_fast"
            }
            
            # Fast JSON writing without indentation
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_result, f, ensure_ascii=False)
            
            analysis_result["analysis_file"] = str(results_path)
            
        except Exception as e:
            logger.error(f"Error saving fast analysis results: {e}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def get_document_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document summary"""
        summary = {
            "file_name": analysis_result.get("file_name", "Unknown"),
            "file_type": analysis_result.get("file_type", "Unknown"),
            "total_pages": len(analysis_result.get("pages", [])),
            "total_tables": len(analysis_result.get("tables", [])),
            "total_images": len(analysis_result.get("images", [])),
            "has_handwriting": len(analysis_result.get("handwriting_text", [])) > 0,
            "processing_errors": len(analysis_result.get("processing_errors", [])),
            "content_types": []
        }
        
        # Determine content types
        if summary["total_pages"] > 0:
            summary["content_types"].append("text")
        if summary["total_tables"] > 0:
            summary["content_types"].append("tables")
        if summary["total_images"] > 0:
            summary["content_types"].append("images")
        if summary["has_handwriting"]:
            summary["content_types"].append("handwriting")
        
        return summary
    
    def list_processed_files(self) -> List[Dict[str, Any]]:
        """List all processed files with their analysis results"""
        processed_files = []
        
        for analysis_file in self.processed_dir.glob("*_analysis.json"):
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis_result = json.load(f)
                
                summary = self.get_document_summary(analysis_result)
                summary["analysis_file"] = str(analysis_file)
                summary["analysis_date"] = analysis_file.stat().st_mtime
                
                processed_files.append(summary)
                
            except Exception as e:
                logger.error(f"Error reading analysis file {analysis_file}: {e}")
        
        return sorted(processed_files, key=lambda x: x.get("analysis_date", 0), reverse=True)