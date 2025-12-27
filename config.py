"""
Stage 1: The Captioner - Configuration Management
Deep-ML Style: Centralized, environment-aware configuration.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LoRAConfig:
    """Mathematical Adapters Configuration."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class TrainingConfig:
    """Training Hyperparameters."""
    # Model
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    
    # Training Schedule
    num_epochs: int = 5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    
    # Precision
    use_4bit: bool = True
    bf16: bool = True
    
    # Checkpointing
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 250

@dataclass
class PathConfig:
    """Path Configuration - Slurm-aware."""
    # Detect if running on Big Red 200
    is_slurm: bool = field(default_factory=lambda: "SLURM_JOB_ID" in os.environ)
    
    # Base paths
    scratch_dir: str = field(default_factory=lambda: 
        os.environ.get("SCRATCH", f"/N/scratch/{os.environ.get('USER', 'user')}")
    )
    
    # Dataset paths
    dataset_dir: str = ""
    annotation_file: str = ""
    
    # Output paths
    output_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    
    def __post_init__(self):
        """Set derived paths after initialization."""
        self.dataset_dir = self.dataset_dir or os.path.join(self.scratch_dir, "phase1_data")
        self.annotation_file = self.annotation_file or os.path.join(self.dataset_dir, "captions.json")
        self.output_dir = self.output_dir or os.path.join(self.scratch_dir, "phase1_output")
        self.checkpoint_dir = self.checkpoint_dir or os.path.join(self.output_dir, "checkpoints")
        self.log_dir = self.log_dir or os.path.join(self.output_dir, "logs")
        
        # Create directories if they don't exist
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(path, exist_ok=True)

@dataclass
class Phase1Config:
    """Master Configuration for Stage 1: The Captioner."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Hardware
    device: str = field(default_factory=lambda: 
        "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    
    # Reproducibility
    seed: int = 42

def get_config(**overrides) -> Phase1Config:
    """Factory function with optional overrides."""
    config = Phase1Config()
    
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.lora, key):
            setattr(config.lora, key, value)
        elif hasattr(config.paths, key):
            setattr(config.paths, key, value)
    
    return config

if __name__ == "__main__":
    # Test configuration
    cfg = get_config()
    print(f"Model: {cfg.training.model_id}")
    print(f"Device: {cfg.device}")
    print(f"Running on Slurm: {cfg.paths.is_slurm}")
    print(f"Output Dir: {cfg.paths.output_dir}")
