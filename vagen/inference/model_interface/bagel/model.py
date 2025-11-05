# vagen/inference/model_interface/bagel/model.py

import logging
import os
from datetime import datetime
import torch
from typing import List, Dict, Any, Union
from PIL import Image

from vagen.inference.model_interface.base_model import BaseModelInterface
from .model_config import BAGELModelConfig

logger = logging.getLogger(__name__)

class BAGELModelInterface(BaseModelInterface):
    """
    Model interface for BAGEL multimodal foundation model.
    Supports understanding (image+text -> text) and generation (text/image -> image).
    """
    
    def __init__(self, config: BAGELModelConfig):
        """Initialize the BAGEL model interface."""
        super().__init__(config.to_dict())
        
        self.config = config
        self.model_path = config.model_path
        self.output_mode = config.output_mode
        
        logger.info(f"Initializing BAGEL model from {self.model_path}")
        logger.info(f"Output mode: {self.output_mode}, Quantization mode: {config.quantization_mode}")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("BAGEL requires CUDA. No GPU detected.")
        
        # Import BAGEL dependencies
        try:
            from accelerate import (
                infer_auto_device_map, 
                load_checkpoint_and_dispatch, 
                init_empty_weights
            )
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
            
            from .inferencer import InterleaveInferencer
            from .data.data_utils import add_special_tokens
            from .data.transforms import ImageTransform
            from .modeling.autoencoder import load_ae
            from .modeling.bagel import (
                BagelConfig, Bagel, 
                Qwen2Config, Qwen2ForCausalLM,
                SiglipVisionConfig, SiglipVisionModel
            )
            from .modeling.qwen2 import Qwen2Tokenizer
            
            self._accelerate_modules = {
                'infer_auto_device_map': infer_auto_device_map,
                'load_checkpoint_and_dispatch': load_checkpoint_and_dispatch,
                'init_empty_weights': init_empty_weights,
                'BnbQuantizationConfig': BnbQuantizationConfig,
                'load_and_quantize_model': load_and_quantize_model
            }
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import BAGEL dependencies. "
                f"Ensure BAGEL is installed: pip install -e /workspace/BAGEL\n"
                f"Error: {e}"
            )
        
        # Load model components
        self._load_model()
        
        logger.info("BAGEL model initialization complete")
    
    def _load_model(self):
        """Load BAGEL model following the app.py workflow."""
        
        # Step 1: Load configurations
        logger.info("Loading model configurations...")
        from modeling.bagel import Qwen2Config, SiglipVisionConfig, BagelConfig
        from modeling.autoencoder import load_ae
        
        llm_config = Qwen2Config.from_json_file(
            os.path.join(self.model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(self.model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        
        vae_model, vae_config = load_ae(
            local_path=os.path.join(self.model_path, "ae.safetensors")
        )
        
        # Step 2: Create BagelConfig
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )
        
        # Step 3: Initialize empty model structure
        logger.info("Creating model structure...")
        from modeling.bagel import Qwen2ForCausalLM, SiglipVisionModel, Bagel
        
        init_empty_weights = self._accelerate_modules['init_empty_weights']
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )
        
        # Step 4: Generate device map
        logger.info("Generating device map...")
        infer_auto_device_map = self._accelerate_modules['infer_auto_device_map']
        
        device_map = infer_auto_device_map(
            model,
            max_memory={i: self.config.max_memory_per_gpu 
                       for i in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        
        # Ensure certain modules are on the same device
        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]
        
        if torch.cuda.device_count() == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device
        
        # Step 5: Load weights based on quantization mode
        logger.info(f"Loading weights (mode {self.config.quantization_mode})...")
        checkpoint_path = os.path.join(self.model_path, "ema.safetensors")
        
        if self.config.quantization_mode == 1:
            # Full precision
            load_checkpoint_and_dispatch = self._accelerate_modules['load_checkpoint_and_dispatch']
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=checkpoint_path,
                device_map=device_map,
                offload_buffers=True,
                offload_folder=self.config.offload_folder,
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()
            
        elif self.config.quantization_mode == 2:
            # NF4 quantization (recommended)
            BnbQuantizationConfig = self._accelerate_modules['BnbQuantizationConfig']
            load_and_quantize_model = self._accelerate_modules['load_and_quantize_model']
            
            bnb_config = BnbQuantizationConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4"
            )
            model = load_and_quantize_model(
                model,
                weights_location=checkpoint_path,
                bnb_quantization_config=bnb_config,
                device_map=device_map,
                offload_folder=self.config.offload_folder,
            ).eval()
            
        elif self.config.quantization_mode == 3:
            # INT8 quantization
            BnbQuantizationConfig = self._accelerate_modules['BnbQuantizationConfig']
            load_and_quantize_model = self._accelerate_modules['load_and_quantize_model']
            
            bnb_config = BnbQuantizationConfig(
                load_in_8bit=True,
                torch_dtype=torch.float32
            )
            model = load_and_quantize_model(
                model,
                weights_location=checkpoint_path,
                bnb_quantization_config=bnb_config,
                device_map=device_map,
                offload_folder=self.config.offload_folder,
            ).eval()
        else:
            raise ValueError(f"Invalid quantization_mode: {self.config.quantization_mode}")
        
        self.model = model
        self.vae_model = vae_model
        
        # Step 6: Initialize tokenizer
        logger.info("Initializing tokenizer...")
        from modeling.qwen2 import Qwen2Tokenizer
        from data.data_utils import add_special_tokens
        
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        
        # Step 7: Setup transforms
        logger.info("Setting up image transforms...")
        from data.transforms import ImageTransform
        
        self.vae_transform = ImageTransform(1024, 512, 16)
        self.vit_transform = ImageTransform(980, 224, 14)
        
        # Step 8: Create inferencer
        logger.info("Creating InterleaveInferencer...")
        from inferencer import InterleaveInferencer
        
        self.inferencer = InterleaveInferencer(
            model=self.model,
            vae_model=self.vae_model,
            tokenizer=self.tokenizer,
            vae_transform=self.vae_transform,
            vit_transform=self.vit_transform,
            new_token_ids=self.new_token_ids,
        )
    
    def generate(self, prompts: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for the given prompts.
        
        Args:
            prompts: List of prompts (message dicts with images/text)
            **kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries with 'text' and 'image' fields
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self._process_single_prompt(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing prompt: {e}", exc_info=True)
                results.append({
                    "text": f"Error: {str(e)}",
                    "image": None,
                    "error": str(e)
                })
        
        return results
    
    def _process_single_prompt(self, prompt: Any, **kwargs) -> Dict[str, Any]:
        """Process a single prompt and return the result."""
        
        # Extract multimodal inputs
        input_list = self._extract_multimodal_inputs(prompt)
        
        # Build generation kwargs
        generation_kwargs = {
            'understanding_output': self.output_mode == 'understand',
            'think': kwargs.get('think', self.config.think_enabled),
            'max_think_token_n': kwargs.get('max_think_tokens', self.config.max_think_tokens),
            'do_sample': kwargs.get('do_sample', self.config.do_sample),
            'text_temperature': kwargs.get('text_temperature', self.config.text_temperature),
        }
        
        # Add generation parameters if not in understand mode
        if self.output_mode != 'understand':
            generation_kwargs.update({
                'cfg_text_scale': kwargs.get('cfg_text_scale', self.config.cfg_text_scale),
                'cfg_img_scale': kwargs.get('cfg_img_scale', self.config.cfg_img_scale),
                'cfg_interval': list(kwargs.get('cfg_interval', self.config.cfg_interval)),
                'timestep_shift': kwargs.get('timestep_shift', self.config.timestep_shift),
                'num_timesteps': kwargs.get('num_timesteps', self.config.num_timesteps),
                'cfg_renorm_min': kwargs.get('cfg_renorm_min', self.config.cfg_renorm_min),
                'cfg_renorm_type': kwargs.get('cfg_renorm_type', self.config.cfg_renorm_type),
                'image_shapes': kwargs.get('image_shapes', self.config.image_shapes),
                'enable_taylorseer': kwargs.get('enable_taylorseer', self.config.enable_taylorseer),
            })
        
        # Run inference with autocast
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output_list = self.inferencer.interleave_inference(
                input_lists=input_list,
                **generation_kwargs
            )
        
        # Parse output
        result = {
            'text': None,
            'image': None,
            'usage': {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        }
        
        for output in output_list:
            if isinstance(output, str):
                result['text'] = output
            elif isinstance(output, Image.Image):
                result['image'] = output
        
        # ===== TEMPORARY: Save generated images to disk =====
        if result['image'] is not None:
            try:
                save_dir = "./bagel_outputs"
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"bagel_{timestamp}.png"
                filepath = os.path.join(save_dir, filename)
                result['image'].save(filepath)
                logger.info(f"Saved generated image to: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to save image: {e}")
        # ===== END TEMPORARY =====
        
        return result
    
    def _extract_multimodal_inputs(self, prompt: Any) -> List[Union[str, Image.Image]]:
        """
        Extract images and text from prompt messages.
        Returns interleaved list: [Image, "text", Image, "text", ...]
        """
        input_list = []
        
        # Handle single message dict
        if isinstance(prompt, dict):
            messages = [prompt]
        else:
            messages = prompt
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            # Extract images from multi_modal_data
            if 'multi_modal_data' in message:
                for key, values in message['multi_modal_data'].items():
                    if 'image' in key.lower():
                        for img in values:
                            if isinstance(img, Image.Image):
                                input_list.append(img)
                            elif isinstance(img, dict) and '__pil_image__' in img:
                                from vagen.server.serial import deserialize_pil_image
                                input_list.append(deserialize_pil_image(img))
            
            # Add text content (remove <image> placeholders)
            text_content = content.replace('<image>', '').strip()
            if text_content:
                input_list.append(text_content)
        
        return input_list
    
    def format_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into a prompt string.
        (For compatibility - BAGEL uses interleaved inputs instead)
        """
        text_parts = []
        
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '').replace('<image>', '')
            
            if role and content.strip():
                text_parts.append(f"{role}: {content}")
        
        return "\n".join(text_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the model."""
        info = super().get_model_info()
        
        info.update({
            "name": self.config.model_name,
            "type": "multimodal",
            "output_mode": self.output_mode,
            "supports_images": True,
            "supports_image_generation": self.output_mode in ['generate'],
            "supports_understanding": True,
            "model_path": self.model_path,
            "quantization_mode": self.config.quantization_mode,
            "config_id": self.config.config_id()
        })
        
        return info