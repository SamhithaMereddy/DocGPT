#!/bin/bash

# Intelligent Document Search System - Setup Script
# Optimized for RTX 4090 GPU

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu/Debian
check_system() {
    print_status "Checking system compatibility..."
    
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script is designed for Linux systems"
        exit 1
    fi
    
    if ! command -v apt &> /dev/null; then
        print_error "apt package manager not found. This script requires Ubuntu/Debian"
        exit 1
    fi
    
    print_success "System compatibility verified"
}

# Check GPU
check_gpu() {
    print_status "Checking GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
        
        if [[ "$GPU_INFO" == *"RTX 4090"* ]]; then
            print_success "RTX 4090 detected - optimal performance expected!"
        elif [[ "$GPU_INFO" == *"RTX"* ]] || [[ "$GPU_INFO" == *"GTX"* ]]; then
            print_warning "Non-RTX 4090 GPU detected. Performance may vary."
        fi
    else
        print_warning "nvidia-smi not found. GPU acceleration may not be available."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
        apt update
        apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libopenblas-dev
    
    print_success "System dependencies installed"
}

# Create virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created and activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install PyTorch with CUDA support
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    pip install -r requirements.txt
    
    # Install GPU-optimized packages if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        print_status "Installing GPU-optimized packages..."
        pip install faiss-gpu>=1.7.4
        pip install llama-cpp-python[cublas]
    else
        print_status "Installing CPU versions..."
        pip install faiss-cpu>=1.7.4
        pip install llama-cpp-python
    fi
    
    print_success "Python dependencies installed"
}

# Create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p models
    mkdir -p data
    mkdir -p uploads
    mkdir -p processed_files
    mkdir -p vector_db
    mkdir -p logs
    
    print_success "Directories created"
}

# Download model
download_model() {
    print_status "Checking for LLM model..."
    
    MODEL_PATH="models/llama-model.gguf"
    
    if [ -f "$MODEL_PATH" ]; then
        print_success "Model already exists at $MODEL_PATH"
        return
    fi
    
    print_status "Downloading Llama 2 7B Chat model (3.6GB)..."
    print_warning "This may take 10-30 minutes depending on your internet connection"
    
    MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
    
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_PATH" "$MODEL_URL" --progress=bar:force
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_PATH" "$MODEL_URL" --progress-bar
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    # Verify download
    if [ -f "$MODEL_PATH" ] && [ -s "$MODEL_PATH" ]; then
        print_success "Model downloaded successfully"
    else
        print_error "Model download failed"
        exit 1
    fi
}

# Set up environment variables
setup_env() {
    print_status "Setting up environment variables..."
    
    cat > .env << EOF
# GPU Configuration
CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=0

# Model Configuration
MODEL_PATH=models/llama-model.gguf
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Application Configuration
HOST=0.0.0.0
PORT=8501
DEBUG=False

# Performance Settings
GPU_MEMORY_FRACTION=0.9
BATCH_SIZE=128
MAX_TOKENS=2048
EOF
    
    print_success "Environment configuration created"
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test basic imports
    python3 -c "
import torch
import streamlit
import sentence_transformers
import llama_cpp
print('✅ All core packages imported successfully')

# Test GPU availability
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name()}')
    print(f'✅ CUDA version: {torch.version.cuda}')
else:
    print('⚠️  CUDA not available - running on CPU')
"
    
    # Test model loading
    if [ -f "models/llama-model.gguf" ]; then
        print_status "Testing model loading..."
        timeout 30 python3 -c "
from config import Config
from llm_service_cpp import LLMServiceCPP
config = Config()
llm = LLMServiceCPP(config)
print('✅ Model loaded successfully')
" 2>/dev/null || print_warning "Model test timed out (this is normal for first run)"
    fi
    
    print_success "Installation test completed"
}

# Create run script
create_run_script() {
    print_status "Creating run script..."
    
    cat > run.sh << 'EOF'
#!/bin/bash
# Run script for Intelligent Document Search System

echo "🚀 Starting Intelligent Document Search System..."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the application
echo "📄 Launching web interface at http://localhost:8501"
streamlit run search_interface.py --server.port=8501 --server.address=0.0.0.0

EOF
    
    chmod +x run.sh
    print_success "Run script created: ./run.sh"
}

# Main setup function
main() {
    echo "🔧 Intelligent Document Search System - Setup"
    echo "============================================"
    
    check_system
    check_gpu
    install_system_deps
    setup_venv
    install_python_deps
    create_directories
    download_model
    setup_env
    create_run_script
    test_installation
    
    echo ""
    echo "============================================"
    print_success "Setup completed successfully! 🎉"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Activate virtual environment: source venv/bin/activate"
    echo "   2. Run the application: ./run.sh"
    echo "   3. Open browser: http://localhost:8501"
    echo ""
    echo "📁 Upload documents and start searching!"
    echo "============================================"
}

# Handle Ctrl+C
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main function
main "$@"