import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from llama_cpp import Llama

class LLMServiceCPP:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """Load the GGUF model using llama-cpp-python with optimized GPU acceleration"""
        try:
            model_path = self.config.LOCAL_LLM_MODEL_PATH
            
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                self.logger.info("Please download a GGUF model file to the models directory")
                self._suggest_model_download()
                return
            
            # Import GPU optimizer
            from gpu_optimizer import gpu_optimizer
            
            # Check if model is already cached
            cached_model = gpu_optimizer.get_cached_model("llama_model", str(model_path))
            if cached_model is not None:
                self.llm = cached_model
                self.logger.info("✅ Using cached LLAMA model from GPU memory")
                return
            
            self.logger.info(f"Loading GGUF model: {model_path}")
            
            # Enable GPU optimizations before model loading
            gpu_optimizer.setup_mixed_precision()
            gpu_optimizer.enable_memory_efficient_attention()
            
            # Get current GPU memory status
            gpu_memory = gpu_optimizer.get_gpu_memory_info()
            
            if gpu_memory:
                self.logger.info(f"GPU Memory before model loading: {gpu_memory['free']:.2f}GB free / {gpu_memory['total']:.2f}GB total")
            
            # Calculate optimal parameters for RTX 4090 (24GB VRAM)
            if gpu_memory:
                free_memory = gpu_memory['free']
                
                # Aggressive GPU utilization for faster processing
                if free_memory > 15:  # >15GB free - use maximum settings
                    n_batch = 2048
                    n_threads = 1  # GPU-focused, minimal CPU threads
                    n_gpu_layers = -1  # All layers on GPU
                    n_ctx = min(self.config.N_CTX, 8192)  # Larger context if possible
                elif free_memory > 10:  # >10GB free
                    n_batch = 1024
                    n_threads = 1
                    n_gpu_layers = -1
                    n_ctx = self.config.N_CTX
                elif free_memory > 5:  # >5GB free
                    n_batch = 512
                    n_threads = 2
                    n_gpu_layers = 32  # Most layers on GPU
                    n_ctx = self.config.N_CTX
                else:  # Limited memory
                    n_batch = 256
                    n_threads = 4
                    n_gpu_layers = 16  # Some layers on GPU
                    n_ctx = min(self.config.N_CTX, 2048)
            else:
                # Fallback settings
                n_batch = 512
                n_threads = 4
                n_gpu_layers = -1
                n_ctx = self.config.N_CTX
            
            self.logger.info(f"Optimized settings: batch_size={n_batch}, gpu_layers={n_gpu_layers}, ctx={n_ctx}")
            
            # Use model caching for faster loading
            def create_llama_model():
                return Llama(
                    model_path=str(model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,  # Reduce log noise
                    n_threads=n_threads,
                    n_batch=n_batch,
                    use_mmap=True,  # Memory mapping for efficiency
                    use_mlock=False,  # Don't lock memory
                    f16_kv=True,  # Half precision key-value cache
                    logits_all=False,  # Only last token logits
                    embedding=False,  # No embeddings needed
                    offload_kqv=True,  # Offload KQV to GPU
                    flash_attn=True,  # Enable flash attention if available
                )
            
            # Load model with caching
            self.llm = gpu_optimizer.get_or_create_model(
                model_type="llama_model",
                model_path=str(model_path),
                create_func=create_llama_model
            )
            
            self.logger.info("GGUF model loaded successfully with maximum GPU acceleration")
            
            # Log final GPU memory usage
            final_memory = gpu_optimizer.get_gpu_memory_info()
            if final_memory:
                self.logger.info(f"GPU Memory after model loading: {final_memory['allocated']:.2f}GB allocated, {final_memory['free']:.2f}GB free")
            
            # Optimized warmup
            try:
                self.logger.info("Warming up model with GPU...")
                warmup_response = self.llm(
                    "<|system|>Test<|user|>Hi<|assistant|>",
                    max_tokens=5,
                    temperature=0.1,
                    stop=["</s>"]
                )
                self.logger.info("Model warmup completed successfully")
            except Exception as warmup_error:
                self.logger.warning(f"Model warmup failed: {warmup_error}")
            
        except Exception as e:
            self.logger.error(f"Failed to load GGUF model: {str(e)}")
            self.logger.info("Please ensure you have a valid GGUF model file")
    
    def _suggest_model_download(self):
        """Suggest model download commands"""
        self.logger.info("\n📥 To download a model, use one of these commands:")
        self.logger.info("For Phi-3 Mini (3.8B):")
        self.logger.info("  wget -O models/phi3-mini.gguf https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf")
        self.logger.info("\nFor Llama 3.1 8B:")
        self.logger.info("  wget -O models/llama3.1-8b.gguf https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        self.logger.info("\nFor smaller model (Llama 3.2 1B):")
        self.logger.info("  wget -O models/llama3.2-1b.gguf https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate response using local GGUF model with retrieved context
        
        Args:
            query: User's search query
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated response with metadata
        """
        if not self.llm:
            return {
                'answer': "LLM model not available. Please check the model configuration.",
                'model': "unavailable",
                'context_used': 0,
                'sources': [],
                'query': query
            }
        
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Create prompt with context and query
            prompt = self._create_rag_prompt(query, context)
            
            # Generate response using llama.cpp with GPU-optimized settings
            response = self.llm(
                prompt,
                max_tokens=min(self.config.MAX_TOKENS, 1024),  # Shorter responses for faster generation
                temperature=self.config.LLM_TEMPERATURE,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["</s>", "Human:", "Assistant:", "\n\n\n", "<|", "###"],
                echo=False,
                stream=False,  # Could enable streaming later
                seed=-1,  # Random seed for variety
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            return {
                'answer': generated_text,
                'model': str(self.config.LOCAL_LLM_MODEL_PATH.name),
                'context_used': len(context_docs),
                'sources': self._extract_sources(context_docs),
                'query': query,
                'tokens_generated': response['usage']['completion_tokens'],
                'tokens_total': response['usage']['total_tokens']
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'answer': f"Sorry, I encountered an error while generating the response: {str(e)}",
                'model': "error",
                'context_used': 0,
                'sources': [],
                'query': query
            }
    
    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        if not context_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            source_info = f"Source {i}: {doc.get('file_name', 'Unknown')}"
            content = doc.get('content', '').strip()
            
            # Limit context length per document to fit in context window
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create specialized RAG prompt template for accurate document retrieval"""
        prompt = f"""<|system|>
You are an expert document analysis assistant. Your job is to provide accurate, detailed answers based ONLY on the provided document context. Follow these guidelines:

1. ACCURACY: Only use information explicitly stated in the provided context
2. CITATIONS: Always reference which source(s) you're using (e.g., "According to Source 1..." or "As mentioned in the Slack documentation...")  
3. COMPLETENESS: Provide comprehensive answers when the context contains relevant information
4. HONESTY: If the context doesn't contain sufficient information to answer the question, clearly state this
5. STRUCTURE: Organize your response clearly with proper paragraphs and bullet points when helpful

<|context|>
{context}

<|user|>
{query}

<|assistant|>
Based on the provided documents, I'll answer your question about "{query}":

"""
        return prompt
    
    def _extract_sources(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context documents"""
        sources = []
        for doc in context_docs:
            source = {
                'file_name': doc.get('file_name', 'Unknown'),
                'file_path': doc.get('file_path', ''),
                'similarity_score': doc.get('similarity_score', 0.0),
                'chunk_id': doc.get('chunk_id', ''),
                'file_type': doc.get('file_type', '')
            }
            sources.append(source)
        
        return sources
    
    def summarize_document(self, document_content: str, file_name: str) -> str:
        """Generate a summary of a document"""
        if not self.llm:
            return "LLM model not available"
        
        try:
            # Truncate content if too long
            max_content_length = 3000
            if len(document_content) > max_content_length:
                document_content = document_content[:max_content_length] + "..."
            
            prompt = f"""<|system|>
Please provide a concise summary of the following document.

<|user|>
Document: {file_name}

Content:
{document_content}

Please summarize this document in 2-3 paragraphs.

<|assistant|>
"""
            
            response = self.llm(
                prompt,
                max_tokens=500,
                temperature=0.3,
                stop=["</s>", "<|user|>", "<|system|>"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            self.logger.error(f"Error summarizing document: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using LLM"""
        if not self.llm:
            return []
        
        try:
            prompt = f"""<|system|>
Extract the most important keywords and phrases from the following text. Return only the keywords, separated by commas.

<|user|>
Text:
{text[:2000]}

Extract the key terms and concepts:

<|assistant|>
"""
            
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.2,
                stop=["</s>", "<|user|>", "<|system|>"]
            )
            
            # Parse keywords from response
            keywords_text = response['choices'][0]['text'].strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            return keywords[:10]  # Return top 10 keywords
            
        except Exception as e:
            self.logger.error(f"Error generating keywords: {str(e)}")
            return []
    
    def check_model_availability(self) -> bool:
        """Check if the LLM model is available and working"""
        if not self.llm:
            return False
        
        try:
            test_response = self.llm(
                "<|system|>Test prompt<|user|>Hello<|assistant|>",
                max_tokens=10,
                temperature=0.1
            )
            return True
        except Exception as e:
            self.logger.error(f"Model availability check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.llm:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.config.LOCAL_LLM_MODEL_PATH),
            "context_size": self.config.N_CTX,
            "gpu_layers": self.config.N_GPU_LAYERS,
            "model_exists": self.config.LOCAL_LLM_MODEL_PATH.exists()
        }