"""
Fast Startup Mode - Optimized for minimal loading time
Skips heavy models and warmup operations for faster Streamlit startup
"""
import logging
import os
import sys

logger = logging.getLogger(__name__)

def is_web_interface_running():
    """Check if the application is running in web interface mode."""
    # Streamlit sets a specific environment variable when running
    return 'STREAMLIT_SERVER_RUNNING' in os.environ or any(
        'streamlit' in arg for arg in sys.argv
    )

def is_command_line_operation():
    """Check if a command-line operation like build or search is being performed."""
    # These arguments indicate a command-line operation that needs the full backend
    cli_args = ['--build', '--search', '--interactive', '--stats']
    return any(arg in sys.argv for arg in cli_args)

# Enable fast startup only for the web interface and not for CLI operations
FAST_STARTUP = (
    os.getenv('FAST_STARTUP', 'true').lower() == 'true' 
    and is_web_interface_running() 
    and not is_command_line_operation()
)

def should_skip_heavy_operations():
    """Check if we should skip heavy operations for fast startup"""
    return FAST_STARTUP

def skip_if_fast_startup(operation_name, heavy_func, fallback_func=None):
    """Decorator to skip heavy operations in fast startup mode"""
    if should_skip_heavy_operations():
        logger.info(f"⚡ Fast startup: Skipping {operation_name}")
        return fallback_func() if fallback_func else None
    else:
        return heavy_func()

def log_fast_startup_status():
    """Log the current startup mode"""
    if FAST_STARTUP:
        logger.info("⚡ FAST STARTUP MODE ENABLED - Heavy models will load on demand")
    else:
        logger.info("🔥 FULL STARTUP MODE - All models will preload")

class FastStartupConfig:
    """Configuration for fast startup optimizations"""
    
    # Skip these heavy operations during startup
    SKIP_MODEL_WARMUP = FAST_STARTUP
    SKIP_COMPREHENSIVE_WARMUP = FAST_STARTUP
    SKIP_HEAVY_MODEL_PRELOAD = FAST_STARTUP
    
    # Lazy load these services
    LAZY_LOAD_DOCUMENT_ANALYZER = FAST_STARTUP  # TrOCR model
    LAZY_LOAD_LLM_SERVICE = FAST_STARTUP        # GGUF model
    
    # Optimization settings
    USE_MINIMAL_WARMUP = FAST_STARTUP
    ENABLE_GPU_OPTIMIZATIONS = True  # Always enable GPU opts
    
    @classmethod
    def get_startup_message(cls):
        if FAST_STARTUP:
            return "⚡ Fast startup enabled - models load on demand for 3-5x faster page loads"
        else:
            return "🔥 Full preload mode - all models loaded upfront"
