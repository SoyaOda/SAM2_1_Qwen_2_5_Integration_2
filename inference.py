"""
Inference script for LISA改 (LISA-Kai) model
Supports both single image and batch inference
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import sys

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LISAInference:
    """
    Inference pipeline for LISA改 model
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained model directory
            device: Device to run inference on
            torch_dtype: Data type for model weights
        """
        self.model_path = Path(model_path)
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load model and components
        self._load_model()
    
    def _load_model(self):
        """Load model, tokenizer, and processor"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load configuration
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = LISAConfig(**config_dict)
        else:
            # Use default config
            self.config = LISAConfig()
        
        # Load tokenizer
        self.tokenizer = prepare_tokenizer_for_lisa(
            model_name=self.config.qwen_model_name,
            seg_token=self.config.seg_token,
            padding_side="left"  # Left padding for generation
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.config.qwen_model_name)
        
        # Load model
        self.model = LISA_Model.from_pretrained(
            str(self.model_path),
            config=self.config,
            torch_dtype=self.torch_dtype
        )
        self.model.set_tokenizer(self.tokenizer)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def process_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Process input image
        
        Args:
            image: Input image (path or PIL Image)
        
        Returns:
            Processed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Process image
        pixel_values = self.processor.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values']
        
        return pixel_values.to(self.device, dtype=self.torch_dtype)
    
    def generate_response(
        self,
        image: Union[str, Path, Image.Image],
        instruction: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate response for given image and instruction
        
        Args:
            image: Input image
            instruction: Text instruction
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            text_response: Generated text response
            mask: Segmentation mask if generated, None otherwise
        """
        # Process image
        pixel_values = self.process_image(image)
        
        # Prepare prompt
        prompt = f"Human: {instruction}\nAssistant:"
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate response using generate_with_masks
        with torch.no_grad():
            outputs = self.model.generate_with_masks(
                input_ids=inputs['input_ids'],
                pixel_values=pixel_values,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
        
        # Extract text and mask
        generated_text = outputs['generated_text'][0]
        masks = outputs['masks'][0]  # List of masks for batch item 0
        
        # Get first mask if any were generated
        mask = masks[0].cpu().numpy() if masks else None
        
        # Clean up text (replace SEG token with [MASK] for display)
        text_response = generated_text.replace(self.config.seg_token, "[MASK]")
        
        return text_response, mask
    
    
    def visualize_result(
        self,
        image: Union[str, Path, Image.Image],
        text_response: str,
        mask: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize inference result
        
        Args:
            image: Original image
            text_response: Generated text
            mask: Segmentation mask
            save_path: Path to save visualization
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        fig, axes = plt.subplots(1, 2 if mask is not None else 1, figsize=(12, 6))
        
        if mask is not None:
            # Show original image
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Show image with mask overlay
            axes[1].imshow(image)
            
            # Create colored mask overlay
            mask_colored = np.zeros((*mask.shape, 4))
            mask_colored[mask] = [1, 0, 0, 0.5]  # Red with 50% opacity
            axes[1].imshow(mask_colored)
            
            axes[1].set_title("Segmentation Result")
            axes[1].axis('off')
        else:
            # Just show image
            axes.imshow(image)
            axes.set_title("Input Image")
            axes.axis('off')
        
        # Add text response
        plt.figtext(0.5, 0.02, f"Response: {text_response}", 
                   ha='center', fontsize=10, wrap=True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run inference with LISA改 model")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--instruction', type=str, required=True,
                       help='Text instruction for the model')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Directory to save outputs')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    pipeline = LISAInference(
        model_path=args.model_path,
        device=args.device
    )
    
    # Run inference
    logger.info(f"Processing image: {args.image}")
    logger.info(f"Instruction: {args.instruction}")
    
    text_response, mask = pipeline.generate_response(
        image=args.image,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    
    logger.info(f"Generated response: {text_response}")
    
    if mask is not None:
        logger.info(f"Segmentation mask generated with shape: {mask.shape}")
        # Save mask
        mask_path = output_dir / "mask.npy"
        np.save(mask_path, mask)
        logger.info(f"Mask saved to {mask_path}")
    
    # Visualize result
    if not args.no_visualization:
        vis_path = output_dir / "visualization.png"
        pipeline.visualize_result(
            image=args.image,
            text_response=text_response,
            mask=mask,
            save_path=str(vis_path)
        )
    
    # Save text response
    response_path = output_dir / "response.txt"
    with open(response_path, 'w') as f:
        f.write(f"Instruction: {args.instruction}\n")
        f.write(f"Response: {text_response}\n")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()