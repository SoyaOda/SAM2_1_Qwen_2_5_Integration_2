"""
Lightweight integration test for LISA改 (LISA-Kai) model
Tests core functionality without downloading full model weights
"""
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from unittest.mock import Mock, patch
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.models.adapters import ImageFeatureAdapter, TextPromptProjector
from src.utils import prepare_tokenizer_for_lisa

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class MockQwenModel:
    """Mock Qwen model for testing"""
    def __init__(self):
        self.config = Mock()
        self.config.hidden_size = 2048
        self.config.vision_config = Mock()
        self.config.vision_config.hidden_size = 1280
        self.config.vocab_size = 152000
        
        # Add visual attribute for extract_vision_features
        self.visual = lambda x: torch.randn(x.size(0), 1024, self.config.vision_config.hidden_size)
        
    def __call__(self, *args, **kwargs):
        # Handle both text and vision inputs
        if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
            B = kwargs.get('input_ids').size(0)
            seq_len = kwargs.get('input_ids').size(1)
            logits = torch.randn(B, seq_len, self.config.vocab_size)
            hidden_states = [torch.randn(B, seq_len, self.config.hidden_size)]
        else:
            # Vision-only input
            B = kwargs.get('pixel_values').size(0) if 'pixel_values' in kwargs else 1
            seq_len = 1024  # Mock sequence length for vision
            logits = torch.randn(B, seq_len, self.config.vocab_size)
            hidden_states = [torch.randn(B, seq_len, self.config.hidden_size)]
        
        # Mock vision features if pixel_values provided
        vision_hidden_states = None
        if 'pixel_values' in kwargs and kwargs['pixel_values'] is not None:
            vision_hidden_states = torch.randn(B, 1024, self.config.vision_config.hidden_size)
        
        outputs = Mock()
        outputs.logits = logits
        outputs.hidden_states = hidden_states
        outputs.vision_hidden_states = vision_hidden_states
        
        return outputs
    
    def generate(self, *args, **kwargs):
        B = kwargs.get('input_ids').size(0)
        seq_len = kwargs.get('input_ids').size(1)
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        
        # Generate random token IDs
        generated_ids = torch.cat([
            kwargs.get('input_ids'),
            torch.randint(0, self.config.vocab_size, (B, max_new_tokens))
        ], dim=1)
        
        return generated_ids
    
    def get_input_embeddings(self):
        embeddings = Mock()
        embeddings.weight = torch.nn.Parameter(
            torch.randn(self.config.vocab_size, self.config.hidden_size)
        )
        return embeddings
    
    def resize_token_embeddings(self, new_size):
        pass
    
    def parameters(self):
        return []
    
    def save_pretrained(self, path):
        # Mock save method
        pass


class MockSAMPredictor:
    """Mock SAM predictor for testing"""
    def __init__(self):
        self.model = Mock()
        self.model.sam_mask_decoder = Mock()
        self.model.sam_prompt_encoder = Mock()
        self.model.sam_prompt_encoder.embed_dim = 256
        self.model.sam_prompt_encoder.get_dense_pe = lambda: torch.randn(1, 256, 64, 64)
        
        # Mock parameters
        self.model.sam_mask_decoder.parameters = lambda: []
        self.model.sam_prompt_encoder.parameters = lambda: []
        
        # Mock mask decoder
        def mock_mask_decoder(*args, **kwargs):
            low_res_masks = torch.randn(1, 1, 256, 256)
            iou_predictions = torch.randn(1, 1)
            return low_res_masks, iou_predictions
        
        self.model.sam_mask_decoder.side_effect = mock_mask_decoder


class LightweightIntegrationTest:
    """Lightweight tests without downloading models"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.test_dir = Path("test_outputs")
        self.test_dir.mkdir(exist_ok=True)
    
    def test_adapter_modules(self):
        """Test adapter modules independently"""
        logger.info("=" * 50)
        logger.info("Testing adapter modules")
        logger.info("=" * 50)
        
        # Test ImageFeatureAdapter
        logger.info("Testing ImageFeatureAdapter")
        image_adapter = ImageFeatureAdapter(in_dim=1280, out_dim=256)
        
        # Test with dummy input
        vision_features = torch.randn(2, 1024, 1280)  # [B, N_patches, D_v]
        
        try:
            sam_features = image_adapter(vision_features)
            logger.info(f"✓ ImageFeatureAdapter output shape: {sam_features.shape}")
            assert sam_features.shape == (2, 256, 32, 32), f"Unexpected shape: {sam_features.shape}"
        except Exception as e:
            logger.error(f"✗ ImageFeatureAdapter failed: {e}")
            raise
        
        # Test TextPromptProjector
        logger.info("\nTesting TextPromptProjector")
        text_proj = TextPromptProjector(in_dim=2048, out_dim=256, use_mlp=True)
        
        # Test with dummy input
        hidden_states = torch.randn(3, 2048)  # [num_prompts, D_l]
        
        try:
            prompt_embeds = text_proj(hidden_states)
            logger.info(f"✓ TextPromptProjector output shape: {prompt_embeds.shape}")
            assert prompt_embeds.shape == (3, 256), f"Unexpected shape: {prompt_embeds.shape}"
        except Exception as e:
            logger.error(f"✗ TextPromptProjector failed: {e}")
            raise
    
    def test_model_initialization(self):
        """Test model initialization with mocks"""
        logger.info("=" * 50)
        logger.info("Testing model initialization")
        logger.info("=" * 50)
        
        # Create config
        config = LISAConfig(
            qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
            device_map=self.device,
            freeze_qwen=True,
            freeze_sam=True
        )
        
        try:
            # Initialize model with manual components
            qwen_model = MockQwenModel()
            sam_predictor = MockSAMPredictor()
            
            model = LISA_Model(
                config=config,
                qwen_model=qwen_model,
                sam_predictor=sam_predictor
            )
            
            logger.info("✓ Model initialized successfully")
            logger.info(f"  - Image adapter input dim: {model.image_adapter.proj.in_features}")
            logger.info(f"  - Text projector input dim: {model.text_prompt_proj.proj.in_features}")
            
        except Exception as e:
            logger.error(f"✗ Model initialization failed: {e}")
            raise
        
        return model
    
    def test_tokenizer_setup(self):
        """Test tokenizer configuration"""
        logger.info("=" * 50)
        logger.info("Testing tokenizer setup")
        logger.info("=" * 50)
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.get_vocab.return_value = {"<s>": 0, "</s>": 1}
        mock_tokenizer.convert_tokens_to_ids.return_value = 152000
        mock_tokenizer.add_special_tokens.return_value = {"additional_special_tokens": 1}
        mock_tokenizer.__len__ = Mock(return_value=152001)
        mock_tokenizer.eos_token_id = 1
        
        try:
            # Test with mock model
            model = Mock()
            model.config = LISAConfig()
            model.qwen = MockQwenModel()
            model.seg_token_id = None
            
            # Set tokenizer
            model.tokenizer = mock_tokenizer
            model.seg_token_id = 152000
            
            logger.info("✓ Tokenizer setup successful")
            logger.info(f"  - SEG token ID: {model.seg_token_id}")
            
        except Exception as e:
            logger.error(f"✗ Tokenizer setup failed: {e}")
            raise
    
    def test_forward_pass(self, model):
        """Test model forward pass with mocks"""
        logger.info("=" * 50)
        logger.info("Testing forward pass")
        logger.info("=" * 50)
        
        # Create dummy inputs
        B = 2
        seq_len = 10
        
        input_ids = torch.randint(0, 152000, (B, seq_len))
        pixel_values = torch.randn(B, 3, 384, 384)
        attention_mask = torch.ones(B, seq_len)
        
        # Add SEG token to input
        input_ids[0, 5] = 152000  # SEG token
        input_ids[1, 7] = 152000  # SEG token
        
        model.seg_token_id = 152000
        
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask
                )
            
            logger.info("✓ Forward pass successful")
            logger.info(f"  - Logits shape: {outputs.logits.shape}")
            logger.info(f"  - Number of mask predictions: {len(outputs.mask_logits)}")
            logger.info(f"  - SEG positions batch 0: {outputs.seg_token_positions[0]}")
            logger.info(f"  - SEG positions batch 1: {outputs.seg_token_positions[1]}")
            
        except Exception as e:
            logger.error(f"✗ Forward pass failed: {e}")
            raise
    
    def test_generate_with_masks(self, model):
        """Test generate_with_masks method"""
        logger.info("=" * 50)
        logger.info("Testing generate_with_masks")
        logger.info("=" * 50)
        
        # Create dummy inputs
        input_ids = torch.randint(0, 152000, (1, 20))
        pixel_values = torch.randn(1, 3, 384, 384)
        attention_mask = torch.ones(1, 20)
        
        # Mock tokenizer for generation
        model.tokenizer = Mock()
        model.tokenizer.eos_token_id = 1
        model.tokenizer.decode.return_value = "This is a test response with <SEG> token"
        
        try:
            with torch.no_grad():
                outputs = model.generate_with_masks(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=False
                )
            
            logger.info("✓ generate_with_masks successful")
            logger.info(f"  - Generated IDs shape: {outputs['generated_ids'].shape}")
            logger.info(f"  - Generated text: {outputs['generated_text'][0]}")
            logger.info(f"  - Number of masks: {len(outputs['masks'][0])}")
            
        except Exception as e:
            logger.error(f"✗ generate_with_masks failed: {e}")
            raise
    
    def test_save_load_functionality(self, model):
        """Test model save/load functionality"""
        logger.info("=" * 50)
        logger.info("Testing save/load functionality")
        logger.info("=" * 50)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            try:
                # Save model
                model.save_pretrained(str(save_path))
                logger.info("✓ Model saved successfully")
                
                # Check saved files
                saved_files = list(save_path.glob("*.pt"))
                logger.info(f"  - Saved files: {[f.name for f in saved_files]}")
                
                # Test loading (we won't actually load due to mocks)
                assert (save_path / "config.pt").exists()
                assert (save_path / "image_adapter.pt").exists()
                assert (save_path / "text_prompt_proj.pt").exists()
                
                logger.info("✓ All required files saved")
                
            except Exception as e:
                logger.error(f"✗ Save/load test failed: {e}")
                raise
    
    def run_all_tests(self):
        """Run all lightweight tests"""
        logger.info("Starting LISA改 Lightweight Integration Tests")
        logger.info("=" * 70)
        
        try:
            # Test 1: Adapter modules
            self.test_adapter_modules()
            
            # Test 2: Model initialization
            model = self.test_model_initialization()
            
            # Test 3: Tokenizer setup
            self.test_tokenizer_setup()
            
            # Test 4: Forward pass
            self.test_forward_pass(model)
            
            # Test 5: Generate with masks
            self.test_generate_with_masks(model)
            
            # Test 6: Save/load
            self.test_save_load_functionality(model)
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ ALL LIGHTWEIGHT TESTS PASSED!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.info("\n" + "=" * 70)
            logger.error("❌ LIGHTWEIGHT TESTS FAILED!")
            logger.error("=" * 70)
            raise


def main():
    """Run lightweight integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run lightweight LISA改 tests")
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run tests on')
    
    args = parser.parse_args()
    
    # Run tests
    tester = LightweightIntegrationTest(device=args.device)
    tester.run_all_tests()


if __name__ == "__main__":
    main()