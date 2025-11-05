# vagen/inference/model_interface/bagel/model_config.py

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from vagen.inference.model_interface.base_model_config import BaseModelConfig

@dataclass
class BAGELModelConfig(BaseModelConfig):
    """Configuration for BAGEL multimodal foundation model."""
    
    provider: str = "bagel"

    # Model path
    model_path: str = "/workspace/VAGEN/vagen/inference/model_interface/bagel/BAGEL/models/BAGEL-7B-MoT"
    
    # Device configuration
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # Quantization mode
    # 1: Full precision (32GB+ VRAM)
    # 2: NF4 quantization (12-32GB VRAM, recommended)
    # 3: INT8 quantization (22-32GB VRAM)
    quantization_mode: int = 2
    
    # Output mode: 'understand' for VQA, 'generate' for T2I/editing
    output_mode: str = "understand"
    
    # Thinking mode parameters
    think_enabled: bool = False
    max_think_tokens: int = 1024
    do_sample: bool = False
    text_temperature: float = 0.3
    
    # Generation parameters (for image generation/editing)
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 2.0
    cfg_interval: Tuple[float, float] = (0.4, 1.0)
    timestep_shift: float = 3.0
    num_timesteps: int = 50
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"  # 'global', 'channel', or 'text_channel'
    image_shapes: Tuple[int, int] = (1024, 1024)
    enable_taylorseer: bool = False
    
    # Multi-GPU settings
    max_memory_per_gpu: str = "80GiB"
    offload_folder: str = "offload"
    
    def config_id(self) -> str:
        """Generate unique identifier for this configuration."""
        mode_suffix = f"_{self.output_mode}"
        if self.think_enabled:
            mode_suffix += "_think"
        quant_suffix = f"_q{self.quantization_mode}"
        return f"BAGEL{mode_suffix}{quant_suffix}"
    
    @staticmethod
    def get_provider_info():
        return {
            "description": "BAGEL multimodal foundation model",
            "supports_multimodal": True,
            "supports_understanding": True,
            "supports_generation": True,
            "supports_editing": True,
            "default_model": "BAGEL-7B-MoT"
        }