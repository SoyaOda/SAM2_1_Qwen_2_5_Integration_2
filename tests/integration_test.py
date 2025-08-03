"""
Integration test for LISA改 (LISA-Kai) model with actual model weights
Tests the complete pipeline from loading to inference with real models
"""
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import tempfile
import shutil
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from transformers import AutoProcessor
from huggingface_hub import hf_hub_download
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class IntegrationTest:
    """
    Complete integration test for LISA改 model
    """
    
    def __init__(self, device: str = "cuda", download_models: bool = True):
        """
        Initialize integration test
        
        Args:
            device: Device to run tests on
            download_models: Whether to download model weights
        """
        self.device = device
        self.download_models = download_models
        self.test_dir = Path("test_outputs")
        self.test_dir.mkdir(exist_ok=True)
        
        # Model names
        self.qwen_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.sam_repo_id = "facebook/sam2.1-hiera-large"
        
    def download_sam_checkpoint(self) -> Path:
        """Download SAM2.1 checkpoint from HuggingFace"""
        logger.info(f"Downloading SAM2.1 checkpoint from {self.sam_repo_id}")
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=self.sam_repo_id,
            filename="sam2.1_hiera_large.pt",
            local_dir=str(checkpoint_dir)
        )
        
        logger.info(f"SAM2.1 checkpoint downloaded to {ckpt_path}")
        return Path(ckpt_path)
    
    def download_test_image(self) -> Image.Image:
        """Download a test image"""
        logger.info("Downloading test image")
        
        # Use a sample image URL
        url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Save for reference
        image.save(self.test_dir / "test_image.jpg")
        
        return image
    
    def create_synthetic_image(self) -> Image.Image:
        """Create a synthetic test image with clear objects"""
        logger.info("Creating synthetic test image")
        
        # Create image with simple shapes
        width, height = 640, 480
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw a red rectangle
        draw.rectangle([100, 100, 300, 250], fill='red', outline='darkred', width=3)
        
        # Draw a blue circle
        draw.ellipse([350, 200, 550, 400], fill='blue', outline='darkblue', width=3)
        
        # Draw a green triangle
        triangle_points = [(150, 350), (250, 450), (50, 450)]
        draw.polygon(triangle_points, fill='green', outline='darkgreen', width=3)
        
        # Add text
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
            draw.text((10, 10), "Test Image", fill='black', font=font)
        except:
            draw.text((10, 10), "Test Image", fill='black')
        
        # Save
        image.save(self.test_dir / "synthetic_image.jpg")
        
        return image
    
    def test_model_loading(self) -> LISA_Model:
        """Test loading the model with actual weights"""
        logger.info("=" * 50)
        logger.info("Testing model loading with actual weights")
        logger.info("=" * 50)
        
        # Configure model
        config = LISAConfig(
            qwen_model_name=self.qwen_model_name,
            sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
            sam_config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            freeze_qwen=True,
            freeze_sam=True,
            use_flash_attention=False  # Disable for testing
        )
        
        # Download SAM checkpoint if needed
        if self.download_models and not Path(config.sam_model_name).exists():
            sam_checkpoint = self.download_sam_checkpoint()
            config.sam_model_name = str(sam_checkpoint)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {config.qwen_model_name}")
        tokenizer = prepare_tokenizer_for_lisa(
            model_name=config.qwen_model_name,
            seg_token=config.seg_token
        )
        
        # Load processor
        logger.info("Loading processor")
        processor = AutoProcessor.from_pretrained(config.qwen_model_name)
        
        # Load model
        logger.info("Loading LISA model")
        try:
            model = LISA_Model(config)
            model.set_tokenizer(tokenizer)
            model = model.to(self.device)
            model.eval()
            
            logger.info("✓ Model loaded successfully")
            logger.info(f"  - Qwen hidden size: {config.qwen_hidden_size}")
            logger.info(f"  - Qwen vision hidden size: {config.qwen_vision_hidden_size}")
            logger.info(f"  - SAM image embedding dim: {config.sam_image_embedding_dim}")
            logger.info(f"  - Device: {self.device}")
            
            return model, tokenizer, processor, config
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    def test_basic_inference(self, model, tokenizer, processor, config, image):
        """Test basic inference without SEG token"""
        logger.info("=" * 50)
        logger.info("Testing basic inference (no segmentation)")
        logger.info("=" * 50)
        
        # Prepare inputs
        instruction = "Describe this image in detail."
        prompt = f"Human: {instruction}\nAssistant:"
        
        # Process image
        pixel_values = processor.image_processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'].to(self.device, dtype=model.dtype)
        
        # Tokenize text
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Run inference
        logger.info("Running model forward pass")
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    pixel_values=pixel_values,
                    attention_mask=inputs['attention_mask']
                )
            
            logger.info("✓ Basic inference successful")
            logger.info(f"  - Logits shape: {outputs.logits.shape}")
            logger.info(f"  - SEG positions: {outputs.seg_token_positions}")
            
            # Generate text
            logger.info("Generating text response")
            generated_ids = model.qwen.generate(
                input_ids=inputs['input_ids'],
                pixel_values=pixel_values,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                do_sample=False
            )
            
            generated_text = tokenizer.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"Generated text: {generated_text[:100]}...")
            
        except Exception as e:
            logger.error(f"✗ Basic inference failed: {e}")
            raise
    
    def test_segmentation_inference(self, model, tokenizer, processor, config, image):
        """Test inference with SEG token generation"""
        logger.info("=" * 50)
        logger.info("Testing segmentation inference")
        logger.info("=" * 50)
        
        # Instructions that should trigger segmentation
        test_cases = [
            "Please segment the red object in this image",
            "Can you show me where the blue circle is?",
            "Point to all the shapes in this image",
        ]
        
        for i, instruction in enumerate(test_cases):
            logger.info(f"\nTest case {i+1}: {instruction}")
            
            prompt = f"Human: {instruction}\nAssistant:"
            
            # Process inputs
            pixel_values = processor.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].to(self.device, dtype=model.dtype)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate with masks
            logger.info("Running generate_with_masks")
            try:
                with torch.no_grad():
                    outputs = model.generate_with_masks(
                        input_ids=inputs['input_ids'],
                        pixel_values=pixel_values,
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                
                logger.info("✓ Generation successful")
                logger.info(f"  - Generated text: {outputs['generated_text'][0]}")
                logger.info(f"  - Number of masks: {len(outputs['masks'][0])}")
                logger.info(f"  - SEG positions: {outputs['seg_positions'][0]}")
                
                # Visualize if masks were generated
                if outputs['masks'][0]:
                    self.visualize_results(
                        image,
                        outputs['generated_text'][0],
                        outputs['masks'][0],
                        save_path=self.test_dir / f"segmentation_result_{i+1}.png"
                    )
                
            except Exception as e:
                logger.error(f"✗ Segmentation inference failed: {e}")
                raise
    
    def test_batch_inference(self, model, tokenizer, processor, config):
        """Test batch inference"""
        logger.info("=" * 50)
        logger.info("Testing batch inference")
        logger.info("=" * 50)
        
        # Create multiple images
        images = [
            self.create_synthetic_image(),
            self.create_synthetic_image()  # Same image for simplicity
        ]
        
        instructions = [
            "Describe the shapes in this image",
            "Please segment the red rectangle"
        ]
        
        # Prepare batch inputs
        prompts = [f"Human: {inst}\nAssistant:" for inst in instructions]
        
        # Process images
        pixel_values = processor.image_processor(
            images=images,
            return_tensors="pt"
        )['pixel_values'].to(self.device, dtype=model.dtype)
        
        # Tokenize texts (with padding)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        logger.info(f"Batch size: {len(images)}")
        logger.info(f"Input shapes: {inputs['input_ids'].shape}")
        
        try:
            with torch.no_grad():
                outputs = model.generate_with_masks(
                    input_ids=inputs['input_ids'],
                    pixel_values=pixel_values,
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True
                )
            
            logger.info("✓ Batch inference successful")
            
            for i in range(len(images)):
                logger.info(f"\nBatch item {i+1}:")
                logger.info(f"  - Instruction: {instructions[i]}")
                logger.info(f"  - Generated: {outputs['generated_text'][i]}")
                logger.info(f"  - Masks: {len(outputs['masks'][i])}")
                
        except Exception as e:
            logger.error(f"✗ Batch inference failed: {e}")
            raise
    
    def test_memory_efficiency(self, model, tokenizer, processor, config):
        """Test memory usage and efficiency"""
        logger.info("=" * 50)
        logger.info("Testing memory efficiency")
        logger.info("=" * 50)
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.info(f"Initial GPU memory: {initial_memory:.2f} GB")
            
            # Run inference
            image = self.create_synthetic_image()
            pixel_values = processor.image_processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].to(self.device, dtype=model.dtype)
            
            prompt = "Describe and segment all objects"
            inputs = tokenizer(
                f"Human: {prompt}\nAssistant:",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate_with_masks(
                    input_ids=inputs['input_ids'],
                    pixel_values=pixel_values,
                    max_new_tokens=100
                )
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            
            logger.info(f"Peak GPU memory: {peak_memory:.2f} GB")
            logger.info(f"Current GPU memory: {current_memory:.2f} GB")
            logger.info(f"Memory increase: {peak_memory - initial_memory:.2f} GB")
    
    def visualize_results(self, image, text, masks, save_path):
        """Visualize segmentation results"""
        n_masks = len(masks)
        fig, axes = plt.subplots(1, n_masks + 1, figsize=(5 * (n_masks + 1), 5))
        
        if n_masks == 0:
            axes.imshow(image)
            axes.set_title("No masks generated")
            axes.axis('off')
        else:
            # Show original
            axes[0].imshow(image)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            # Show each mask
            for i, mask in enumerate(masks):
                axes[i+1].imshow(image)
                
                # Overlay mask
                mask_np = mask.cpu().numpy()
                masked = np.ma.masked_where(mask_np < 0.5, mask_np)
                axes[i+1].imshow(masked, alpha=0.5, cmap='Reds')
                
                axes[i+1].set_title(f"Mask {i+1}")
                axes[i+1].axis('off')
        
        plt.suptitle(f"Response: {text[:100]}...", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {save_path}")
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting LISA改 Integration Tests")
        logger.info("=" * 70)
        
        try:
            # Test 1: Model loading
            model, tokenizer, processor, config = self.test_model_loading()
            
            # Test 2: Basic inference with real image
            if self.download_models:
                real_image = self.download_test_image()
                self.test_basic_inference(model, tokenizer, processor, config, real_image)
            
            # Test 3: Segmentation with synthetic image
            synthetic_image = self.create_synthetic_image()
            self.test_segmentation_inference(model, tokenizer, processor, config, synthetic_image)
            
            # Test 4: Batch inference
            self.test_batch_inference(model, tokenizer, processor, config)
            
            # Test 5: Memory efficiency
            self.test_memory_efficiency(model, tokenizer, processor, config)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ ALL INTEGRATION TESTS PASSED!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error("\n" + "=" * 70)
            logger.error("❌ INTEGRATION TESTS FAILED!")
            logger.error("=" * 70)
            raise


def main():
    """Run integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LISA改 integration tests")
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run tests on')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip downloading model weights')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run tests
    tester = IntegrationTest(
        device=args.device,
        download_models=not args.no_download
    )
    
    tester.run_all_tests()


if __name__ == "__main__":
    main()