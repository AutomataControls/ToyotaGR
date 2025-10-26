"""
Optimized Training Configuration for A100 80GB GPU
ORIS - OLYMPUS Racing Intelligence System
"""

import torch

# GPU Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIXED_PRECISION = True  # Use AMP for A100 tensor cores
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size multiplier

# Batch Sizes (optimized for A100 80GB)
BATCH_SIZES = {
    'minerva': 256,      # Strategic analysis can handle large batches
    'atlas': 192,        # Spatial processing moderate batch
    'iris': 224,         # Visual processing 
    'chronos': 320,      # Temporal sequences can be batched heavily
    'prometheus': 128,   # Larger model, smaller batch
    'ensemble': 64       # Ensemble processes all models
}

# Training Epochs (scaled up for production training)
EPOCHS = {
    'minerva': 200,
    'atlas': 150,
    'iris': 175,
    'chronos': 150,
    'prometheus': 250,  # Complex model needs more epochs
    'ensemble': 100
}

# Learning Rates (with warmup)
LEARNING_RATES = {
    'minerva': 5e-4,
    'atlas': 5e-4,
    'iris': 5e-4,
    'chronos': 5e-4,
    'prometheus': 2e-4,  # Lower for stability
    'ensemble': 1e-4
}

# Optimizer Settings
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
GRADIENT_CLIP_VAL = 1.0

# DataLoader Settings
NUM_WORKERS = 8  # Parallel data loading
PIN_MEMORY = True  # Faster GPU transfer
PREFETCH_FACTOR = 2

# Training Strategy
WARMUP_STEPS = 1000
SCHEDULER_TYPE = 'cosine'  # Cosine annealing with warmup
SAVE_FREQUENCY = 10  # Save checkpoint every N epochs

# Memory Optimization
GRADIENT_CHECKPOINTING = True  # Trade compute for memory
EMPTY_CACHE_FREQUENCY = 50  # Clear cache every N batches

# Validation
VAL_BATCH_SIZE_MULTIPLIER = 2  # Double batch size for validation
VAL_FREQUENCY = 5  # Validate every N epochs

# A100 Specific Optimizations
TF32_ENABLED = True  # Enable TF32 for A100
CUDNN_BENCHMARK = True  # Auto-tune convolutions

def get_training_config(model_name: str) -> dict:
    """Get optimized training configuration for specific model"""
    return {
        'batch_size': BATCH_SIZES.get(model_name, 128),
        'epochs': EPOCHS.get(model_name, 100),
        'learning_rate': LEARNING_RATES.get(model_name, 1e-4),
        'device': DEVICE,
        'mixed_precision': MIXED_PRECISION,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'num_workers': NUM_WORKERS,
        'pin_memory': PIN_MEMORY,
        'gradient_clip_val': GRADIENT_CLIP_VAL,
        'weight_decay': WEIGHT_DECAY,
        'warmup_steps': WARMUP_STEPS
    }

def setup_a100_optimization():
    """Configure PyTorch for optimal A100 performance"""
    if torch.cuda.is_available():
        # Enable TF32 for A100
        torch.backends.cuda.matmul.allow_tf32 = TF32_ENABLED
        torch.backends.cudnn.allow_tf32 = TF32_ENABLED
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
        
        # Set memory fraction to prevent OOM with large batches
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        print(f"✅ A100 optimizations enabled")
        print(f"   • TF32: {TF32_ENABLED}")
        print(f"   • cuDNN benchmark: {CUDNN_BENCHMARK}")
        print(f"   • Mixed precision: {MIXED_PRECISION}")