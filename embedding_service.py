import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging
from pathlib import Path
import pickle

class EmbeddingService:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = logging.getLogger(__name__)
        self._query_cache = {}  # Cache for query embeddings
        self._load_model()
    
    def _load_model(self):
        """Load the BGE embedding model with optimized GPU utilization"""
        try:
            self.logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            
            # Check for CUDA availability and use RTX 4090 optimally
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Using device: {device}")
            
            if device == 'cuda':
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                # GPU optimizations for RTX 4090
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Enable memory efficient attention
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Load model directly on target device for efficiency
            try:
                self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device=device)
                
                if device == 'cuda':
                    # Optimize model for RTX 4090
                    self.model.max_seq_length = min(8192, self.model.max_seq_length)
                    # Set model to half precision for memory efficiency
                    self.model = self.model.half()
                    
                self.logger.info(f"Embedding model loaded successfully on {device}")
                
            except Exception as direct_load_error:
                self.logger.warning(f"Direct GPU loading failed: {direct_load_error}")
                # Fallback: Load on CPU then move
                self.model = SentenceTransformer(self.config.EMBEDDING_MODEL, device='cpu')
                if device == 'cuda':
                    self.model = self.model.to(device).half()
                    
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to a smaller model if BGE fails
            self.logger.info("Falling back to all-MiniLM-L6-v2")
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
                if device == 'cuda':
                    self.model = self.model.half()
                self.logger.info(f"Fallback model loaded successfully on {device}")
            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {fallback_error}")
                # Last resort - CPU only
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                self.logger.info("Using CPU-only fallback model")
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks with maximum GPU acceleration
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of documents with embeddings added
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        from gpu_optimizer import gpu_optimizer
        
        # Monitor initial GPU memory
        if torch.cuda.is_available():
            initial_memory = gpu_optimizer.get_gpu_memory_info()
            self.logger.info(f"Starting embedding generation - GPU Memory: {initial_memory['free']:.2f}GB free")
        
        texts = [doc['content'] for doc in documents]
        
        # Pre-allocate and optimize tokenization for GPU
        truncated_texts = []
        tokenizer = self.model.tokenizer
        
        # Batch tokenization for better GPU utilization
        batch_size_tokenize = 32
        for i in range(0, len(texts), batch_size_tokenize):
            batch_texts = texts[i:i + batch_size_tokenize]
            
            # Tokenize batch
            encoded_batch = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=510,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # Decode back to text
            for tokens in encoded_batch['input_ids']:
                truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                truncated_texts.append(truncated_text)
        
        self.logger.info(f"Generating embeddings for {len(truncated_texts)} documents with GPU acceleration")
        
        # Aggressive batch size optimization for RTX 4090
        base_batch_size = 128 if torch.cuda.is_available() else 16
        batch_size = gpu_optimizer.optimize_batch_size(base_batch_size)
        
        # Override with larger batches if we have lots of free memory
        if torch.cuda.is_available():
            memory_info = gpu_optimizer.get_gpu_memory_info()
            if memory_info and memory_info['free'] > 15:  # >15GB free
                batch_size = min(256, len(truncated_texts))  # Very large batches
            elif memory_info and memory_info['free'] > 10:  # >10GB free
                batch_size = min(192, len(truncated_texts))
        
        self.logger.info(f"Using optimized batch size: {batch_size}")
        embeddings = []
        
        # Pre-compile model for faster inference (if not already done)
        if torch.cuda.is_available() and hasattr(torch, 'compile'):
            try:
                if not hasattr(self.model, '_is_compiled'):
                    self.model = torch.compile(self.model, mode='max-autotune')
                    self.model._is_compiled = True
                    self.logger.info("Model compiled for maximum speed")
            except Exception as e:
                self.logger.debug(f"Model compilation not available: {e}")
        
        for i in range(0, len(truncated_texts), batch_size):
            batch_texts = truncated_texts[i:i + batch_size]
            
            # Use mixed precision with aggressive GPU optimization
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    # Enable CUDA graph optimization for consistent batch sizes
                    with torch.backends.cudnn.flags(enabled=True, benchmark=True):
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            convert_to_tensor=True,  # Keep on GPU initially
                            show_progress_bar=(i == 0),
                            normalize_embeddings=True,
                            batch_size=batch_size,
                            device='cuda'  # Ensure GPU usage
                        )
                    
                    # Convert to CPU numpy for storage (batch conversion is faster)
                    batch_embeddings = batch_embeddings.cpu().numpy()
            else:
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=(i == 0),
                        normalize_embeddings=True,
                        batch_size=batch_size
                    )
            
            embeddings.extend(batch_embeddings)
            
            # Aggressive memory monitoring and cleanup
            if torch.cuda.is_available():
                if i % (batch_size * 2) == 0:  # More frequent monitoring
                    gpu_optimizer.monitor_memory_usage(threshold_gb=18.0)  # Lower threshold for aggressive cleanup
        
        # Add embeddings to documents with optimized memory usage
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            doc['embedding_model'] = self.config.EMBEDDING_MODEL
        
        # Final memory status
        if torch.cuda.is_available():
            final_memory = gpu_optimizer.get_gpu_memory_info()
            self.logger.info(f"Embeddings generated successfully - Final GPU Memory: {final_memory['allocated']:.2f}GB allocated, {final_memory['free']:.2f}GB free")
        
        return documents
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query with caching and GPU optimization
        
        Args:
            query: Search query string
            
        Returns:
            Query embedding as numpy array
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Check cache first for instant results
        query_hash = hash(query.strip().lower())
        if query_hash in self._query_cache:
            self.logger.debug("Using cached embedding for query")
            return self._query_cache[query_hash]
        
        # Generate embedding with GPU optimization
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                embedding = self.model.encode(
                    query,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,  # Disable for single queries
                    batch_size=1
                )
        else:
            with torch.no_grad():
                embedding = self.model.encode(
                    query,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1
                )
        
        # Cache the result (limit cache size to prevent memory issues)
        if len(self._query_cache) < 100:  # Keep last 100 queries
            self._query_cache[query_hash] = embedding
        elif len(self._query_cache) >= 100:
            # Remove oldest entry (simple cleanup)
            self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[query_hash] = embedding
        
        return embedding
    
    def save_embeddings(self, documents: List[Dict[str, Any]], file_path: str):
        """Save embeddings to disk"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(documents, f)
            self.logger.info(f"Embeddings saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {str(e)}")
    
    def load_embeddings(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embeddings from disk"""
        try:
            with open(file_path, 'rb') as f:
                documents = pickle.load(f)
            self.logger.info(f"Embeddings loaded from {file_path}")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {str(e)}")
            return []
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for doc_embedding in doc_embeddings:
            # Ensure embeddings are numpy arrays
            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)
            
            # Compute cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append(float(similarity))
        
        return similarities