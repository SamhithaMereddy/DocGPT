# 🔍 Intelligent Document Search System

An AI-powered document search system optimized for RTX 4090 GPU with local LLM processing, semantic search, and fast startup capabilities.

## ✨ Features

- 🚀 **Fast Startup**: 3-5x faster loading with on-demand model initialization
- 🎯 **Semantic Search**: Advanced document understanding using BGE embeddings
- 🖥️ **GPU Optimized**: Specifically tuned for RTX 4090 performance
- 📄 **Multi-Format Support**: PDF, DOCX, TXT, HTML, Markdown
- 🤖 **Local LLM**: Llama 2 7B Chat with GGUF quantization
- 🔒 **Privacy First**: All processing runs locally, no data leaves your machine
- 📊 **Smart Analytics**: Document insights and search performance metrics
- 🌐 **Web Interface**: Clean, responsive Streamlit interface

## 🎯 Optimized For

- **Hardware**: RTX 4090 (24GB VRAM)
- **Performance**: 140+ tokens/second inference
- **Memory**: Efficient 4-bit quantization (3.6GB model)
- **Context**: 8K token context window

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/indra7777/document-search-system.git
cd document-search-system

# Run automated setup (installs everything)
./setup.sh

# Start the application
./run.sh
```

### Option 2: Manual Setup

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv build-essential cmake

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install requirements
pip install -r requirements.txt

# 5. Install GPU-optimized packages
pip install faiss-gpu llama-cpp-python[cublas]

# 6. Download the model (3.6GB)
mkdir -p models
wget -O models/llama-model.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf

# 7. Run the application
streamlit run search_interface.py
```

## 📋 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / Debian 11+
- **Python**: 3.8+
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space
- **GPU**: CUDA-compatible (optional but recommended)

### Recommended (RTX 4090)
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **RAM**: 32GB system RAM
- **GPU**: RTX 4090 (24GB VRAM)
- **Storage**: 20GB free space (SSD preferred)
- **CUDA**: 11.8+

## 🔧 Configuration

### Environment Variables (.env)
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9

# Model Settings
MODEL_PATH=models/llama-model.gguf
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Application
HOST=0.0.0.0
PORT=8501
DEBUG=False
```

### Advanced Configuration (config.py)
- **Chunk Size**: Document processing parameters
- **Embedding Dimensions**: Vector database settings  
- **GPU Memory**: CUDA optimization settings
- **Search Parameters**: Similarity thresholds and result limits

## 📖 Usage

### 1. Start the Application
```bash
./run.sh
# Or manually: streamlit run search_interface.py
```

### 2. Upload Documents
- Navigate to http://localhost:8501
- Use the sidebar to upload PDF, DOCX, TXT files
- Wait for processing completion

### 3. Search Documents
- Enter natural language queries
- Get AI-powered responses with source citations
- View document chunks and similarity scores

### 4. Advanced Features
- **Fast Startup Mode**: Toggle for 3-5x faster loading
- **GPU Warmup**: Pre-allocate GPU memory for consistent performance
- **Batch Processing**: Upload multiple documents simultaneously

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Processing     │───▶│   Vector DB     │
│  (PDF, DOCX)    │    │  (Chunking +     │    │   (FAISS)       │
└─────────────────┘    │   Embedding)     │    └─────────────────┘
                       └──────────────────┘             │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │◀───│   Search Engine  │◀───│   Retrieval     │
│ (Streamlit)     │    │   (LLM + RAG)    │    │   (Semantic)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

- **Document Processor**: Handles multiple formats with optimized chunking
- **Embedding Service**: BGE-large-en-v1.5 for semantic understanding
- **Vector Database**: FAISS with GPU acceleration
- **LLM Service**: Llama 2 7B Chat with GGUF quantization
- **Service Manager**: Memory-efficient service lifecycle management
- **GPU Optimizer**: RTX 4090-specific optimizations

## 🔍 Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Text extraction, image recognition |
| Word | `.docx` | Full document structure |
| Text | `.txt` | Direct text processing |
| HTML | `.html` | Web content parsing |
| Markdown | `.md` | Structured text processing |

## ⚡ Performance Benchmarks

### RTX 4090 Performance
- **Model Loading**: ~6 seconds
- **Document Processing**: ~500 pages/minute
- **Embedding Generation**: ~1000 chunks/minute
- **Search Latency**: <100ms per query
- **LLM Inference**: 140+ tokens/second

### Memory Usage
- **Model**: 3.6GB GPU memory
- **KV Cache**: 4GB GPU memory  
- **Embeddings**: ~1GB system RAM (per 1000 documents)
- **Vector DB**: ~500MB GPU memory

## 🛠️ Development

### Project Structure
```
document-search-system/
├── search_interface.py      # Main Streamlit application
├── document_analyzer.py     # Document processing engine
├── embedding_service.py     # Semantic embedding generation
├── vector_database.py       # FAISS vector storage
├── llm_service_cpp.py      # Llama.cpp LLM interface
├── service_manager.py       # Resource management
├── gpu_optimizer.py         # RTX 4090 optimizations
├── fast_startup.py          # Quick initialization
├── config.py               # Configuration management
├── setup.py                # Package installation
├── requirements.txt         # Python dependencies
└── setup.sh                # Automated setup script
```

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Model Download Fails**
```bash
# Manual download
wget -O models/llama-model.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
```

**Out of Memory Errors**
- Reduce `GPU_MEMORY_FRACTION` in config.py
- Lower `CUDA_BATCH_SIZE` for embedding processing
- Enable `USE_MIXED_PRECISION` for memory savings

**Slow Performance**
- Ensure GPU drivers are up to date
- Check `nvidia-smi` for GPU utilization
- Verify model is loading on GPU (check logs)

### Debug Mode
```bash
# Enable detailed logging
export DEBUG=True
streamlit run search_interface.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Llama 2**: Meta's foundation model
- **BGE Embeddings**: BAAI's semantic understanding
- **FAISS**: Facebook's similarity search
- **Streamlit**: Web interface framework
- **llama.cpp**: Efficient LLM inference

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/indra7777/document-search-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/indra7777/document-search-system/discussions)
- **Email**: y21co021@rvrjc.ac.in

## 🚀 Roadmap

- [ ] Support for additional LLM models
- [ ] Multi-language document processing
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment options
- [ ] Advanced analytics dashboard
- [ ] Real-time document monitoring

---

**Made with ❤️ for RTX 4090 users**

*Optimized for performance, built for productivity*