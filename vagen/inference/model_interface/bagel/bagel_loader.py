import sys
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BAGELAccelerateLoader:
    """Loader for BAGEL modules that require accelerate and model components"""
    
    def __init__(self, bagel_path: str = "/workspace/BAGEL"):
        self.bagel_path = Path(bagel_path)
        if not self.bagel_path.exists():
            raise FileNotFoundError(f"BAGEL path does not exist: {self.bagel_path}")
        
        if str(self.bagel_path) not in sys.path:
            sys.path.insert(0, str(self.bagel_path))
            logger.info(f"Added BAGEL path to sys.path: {self.bagel_path}")
    
    def load_accelerate_modules(self) -> Dict[str, Any]:
        """Load accelerate-related modules"""
        modules = {}
        
        try:
            logger.info("Loading accelerate modules...")
            from accelerate import (
                infer_auto_device_map, 
                load_checkpoint_and_dispatch, 
                init_empty_weights
            )
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
            
            modules['infer_auto_device_map'] = infer_auto_device_map
            modules['load_checkpoint_and_dispatch'] = load_checkpoint_and_dispatch
            modules['init_empty_weights'] = init_empty_weights
            modules['BnbQuantizationConfig'] = BnbQuantizationConfig
            modules['load_and_quantize_model'] = load_and_quantize_model
            
            logger.info("Successfully loaded accelerate modules")
            return modules
            
        except ImportError as e:
            raise ImportError(
                f"Failed to load accelerate modules. "
                f"Make sure accelerate is installed: {e}"
            )
    
    def load_bagel_modules(self) -> Dict[str, Any]:
        """Load BAGEL-specific modules (inferencer, data utils, modeling)"""
        modules = {}
        
        try:
            # Load inferencer
            logger.info("Loading BAGEL inferencer...")
            from inferencer import InterleaveInferencer
            modules['InterleaveInferencer'] = InterleaveInferencer
            
            # Load data utilities
            logger.info("Loading BAGEL data utilities...")
            from data.data_utils import add_special_tokens
            from data.transforms import ImageTransform
            modules['add_special_tokens'] = add_special_tokens
            modules['ImageTransform'] = ImageTransform
            
            # Load autoencoder
            logger.info("Loading BAGEL autoencoder...")
            from modeling.autoencoder import load_ae
            modules['load_ae'] = load_ae
            
            # Load BAGEL modeling components
            logger.info("Loading BAGEL modeling components...")
            from modeling.bagel import (
                BagelConfig, Bagel, 
                Qwen2Config, Qwen2ForCausalLM,
                SiglipVisionConfig, SiglipVisionModel
            )
            modules['BagelConfig'] = BagelConfig
            modules['Bagel'] = Bagel
            modules['Qwen2Config'] = Qwen2Config
            modules['Qwen2ForCausalLM'] = Qwen2ForCausalLM
            modules['SiglipVisionConfig'] = SiglipVisionConfig
            modules['SiglipVisionModel'] = SiglipVisionModel
            
            # Load tokenizer
            logger.info("Loading Qwen2 tokenizer...")
            from modeling.qwen2 import Qwen2Tokenizer
            modules['Qwen2Tokenizer'] = Qwen2Tokenizer
            
            logger.info("Successfully loaded all BAGEL modules")
            return modules
            
        except ImportError as e:
            raise ImportError(
                f"Failed to load BAGEL modules from {self.bagel_path}. "
                f"Error: {e}\n"
                f"Make sure BAGEL repository is properly set up and all dependencies are installed."
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while loading BAGEL modules: {e}"
            )
    
    def load_all_modules(self) -> Dict[str, Any]:
        """Load both accelerate and BAGEL modules"""
        all_modules = {}
        
        # Load accelerate modules
        accelerate_modules = self.load_accelerate_modules()
        all_modules.update(accelerate_modules)
        
        # Load BAGEL modules
        bagel_modules = self.load_bagel_modules()
        all_modules.update(bagel_modules)
        
        logger.info(f"Successfully loaded {len(all_modules)} modules in total")
        return all_modules