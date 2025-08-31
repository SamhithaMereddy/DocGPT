"""
GPU-Accelerated Text Processing for Document Analysis
Provides GPU tensor operations for faster text preprocessing and analysis
"""
import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer
import logging
from gpu_optimizer import gpu_optimizer

logger = logging.getLogger(__name__)

class GPUTextProcessor:
    """GPU-accelerated text processing for document analysis"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Initialize tokenizer for text processing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Setup GPU optimizations
        if torch.cuda.is_available():
            gpu_optimizer.setup_mixed_precision()
            
        logger.info(f"GPU Text Processor initialized on {self.device}")
    
    def batch_tokenize_gpu(self, texts: List[str], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        GPU-accelerated batch tokenization
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized tensors on GPU
        """
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer not available")
            
            # Batch tokenization
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to GPU
            gpu_encoded = {}
            for key, tensor in encoded.items():
                gpu_encoded[key] = tensor.to(self.device)
            
            return gpu_encoded
            
        except Exception as e:
            logger.error(f"Error in GPU tokenization: {e}")
            raise
    
    def compute_text_similarity_gpu(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise text similarity using GPU tensor operations
        
        Args:
            text_embeddings: Tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Similarity matrix of shape (batch_size, batch_size)
        """
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Normalize embeddings
                embeddings_norm = F.normalize(text_embeddings, p=2, dim=1)
                
                # Compute cosine similarity matrix
                similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
                
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing text similarity: {e}")
            return torch.zeros((text_embeddings.shape[0], text_embeddings.shape[0]), 
                             device=self.device)
    
    def extract_keywords_gpu(self, text_tensor: torch.Tensor, 
                           attention_weights: torch.Tensor, 
                           top_k: int = 10) -> List[int]:
        """
        Extract keywords using GPU-accelerated attention weights
        
        Args:
            text_tensor: Tokenized text tensor
            attention_weights: Attention weights from transformer model
            top_k: Number of top keywords to extract
            
        Returns:
            List of token indices representing keywords
        """
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Average attention across heads and layers
                if attention_weights.dim() > 2:
                    avg_attention = torch.mean(attention_weights, dim=(0, 1))
                else:
                    avg_attention = attention_weights
                
                # Get top-k tokens with highest attention
                _, top_indices = torch.topk(avg_attention, top_k)
                
            return top_indices.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def batch_text_classification_gpu(self, text_features: torch.Tensor, 
                                    classification_weights: torch.Tensor,
                                    classification_bias: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated batch text classification
        
        Args:
            text_features: Text feature tensors (batch_size, feature_dim)
            classification_weights: Classification layer weights
            classification_bias: Classification layer bias
            
        Returns:
            Classification logits
        """
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Linear classification layer
                logits = F.linear(text_features, classification_weights, classification_bias)
                
                # Apply softmax for probabilities
                probabilities = F.softmax(logits, dim=-1)
                
            return probabilities
            
        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            return torch.zeros((text_features.shape[0], classification_weights.shape[0]), 
                             device=self.device)
    
    def chunk_text_gpu(self, text: str, chunk_size: int = 512, 
                      overlap: int = 50) -> List[str]:
        """
        GPU-optimized text chunking with overlap
        
        Args:
            text: Input text string
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        try:
            # Convert to tensor for GPU operations
            text_bytes = text.encode('utf-8')
            text_tensor = torch.frombuffer(text_bytes, dtype=torch.uint8).to(self.device)
            
            chunks = []
            start = 0
            
            while start < len(text_tensor):
                end = min(start + chunk_size, len(text_tensor))
                
                # Extract chunk
                chunk_tensor = text_tensor[start:end]
                
                # Convert back to string
                chunk_bytes = chunk_tensor.cpu().numpy().tobytes()
                try:
                    chunk_text = chunk_bytes.decode('utf-8')
                    chunks.append(chunk_text)
                except UnicodeDecodeError:
                    # Handle incomplete UTF-8 sequences at boundaries
                    valid_end = end
                    while valid_end > start:
                        try:
                            chunk_bytes = text_tensor[start:valid_end].cpu().numpy().tobytes()
                            chunk_text = chunk_bytes.decode('utf-8')
                            chunks.append(chunk_text)
                            break
                        except UnicodeDecodeError:
                            valid_end -= 1
                    end = valid_end
                
                start = end - overlap
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error in GPU text chunking: {e}")
            # Fallback to CPU chunking
            return self._chunk_text_cpu(text, chunk_size, overlap)
    
    def _chunk_text_cpu(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Fallback CPU text chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
            
        return chunks
    
    def detect_language_gpu(self, text_features: torch.Tensor, 
                          language_classifier_weights: torch.Tensor) -> torch.Tensor:
        """
        GPU-accelerated language detection
        
        Args:
            text_features: Text feature tensors
            language_classifier_weights: Language classification weights
            
        Returns:
            Language prediction probabilities
        """
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Simple linear classification for language detection
                language_logits = torch.matmul(text_features, language_classifier_weights.t())
                language_probs = F.softmax(language_logits, dim=-1)
                
            return language_probs
            
        except Exception as e:
            logger.error(f"Error in language detection: {e}")
            return torch.zeros((text_features.shape[0], language_classifier_weights.shape[0]), 
                             device=self.device)
    
    def batch_regex_search_gpu(self, texts: List[str], patterns: List[str]) -> List[List[Dict[str, Any]]]:
        """
        GPU-accelerated batch regex search (CPU fallback for regex)
        
        Args:
            texts: List of text strings
            patterns: List of regex patterns
            
        Returns:
            List of matches for each text
        """
        try:
            all_matches = []
            
            # Compile patterns for efficiency
            compiled_patterns = [re.compile(pattern) for pattern in patterns]
            
            for text in texts:
                text_matches = []
                for i, pattern in enumerate(compiled_patterns):
                    matches = pattern.finditer(text)
                    for match in matches:
                        text_matches.append({
                            'pattern_index': i,
                            'pattern': patterns[i],
                            'match': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                            'groups': match.groups()
                        })
                all_matches.append(text_matches)
            
            return all_matches
            
        except Exception as e:
            logger.error(f"Error in regex search: {e}")
            return [[] for _ in texts]
    
    def compute_text_stats_gpu(self, tokenized_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute text statistics using GPU tensor operations
        
        Args:
            tokenized_batch: Batch of tokenized texts
            
        Returns:
            Dictionary of text statistics
        """
        try:
            input_ids = tokenized_batch['input_ids']
            attention_mask = tokenized_batch['attention_mask']
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # Sequence lengths
                seq_lengths = attention_mask.sum(dim=1)
                
                # Token frequency statistics
                vocab_size = input_ids.max() + 1
                token_counts = torch.zeros(vocab_size, device=self.device)
                
                for seq in input_ids:
                    unique_tokens, counts = torch.unique(seq, return_counts=True)
                    token_counts.scatter_add_(0, unique_tokens, counts.float())
                
                # Compute diversity metrics
                unique_tokens_per_seq = []
                for seq, mask in zip(input_ids, attention_mask):
                    valid_tokens = seq[mask.bool()]
                    unique_count = len(torch.unique(valid_tokens))
                    unique_tokens_per_seq.append(unique_count)
                
                unique_tokens_tensor = torch.tensor(unique_tokens_per_seq, device=self.device)
                
            stats = {
                'sequence_lengths': seq_lengths,
                'token_counts': token_counts,
                'unique_tokens_per_sequence': unique_tokens_tensor,
                'avg_sequence_length': seq_lengths.float().mean(),
                'avg_unique_tokens': unique_tokens_tensor.float().mean()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error computing text stats: {e}")
            return {}
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage"""
        return gpu_optimizer.get_gpu_memory_info()
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        gpu_optimizer.cleanup_gpu_memory(force=True)

# Global GPU text processor instance
gpu_text_processor = GPUTextProcessor()