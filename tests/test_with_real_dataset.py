"""
Test LISA model with real dataset format
"""
import sys
import torch
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import LISAConfig
from src.models import LISA_Model
from src.utils import prepare_tokenizer_for_lisa
from src.data.dataset import HybridDataset, collate_fn
from src.data.collators import MultiModalDataCollator
from transformers import AutoProcessor
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_model_integration():
    """Test that model can process data from actual datasets"""
    
    # 1. Setup configuration
    config = LISAConfig(
        qwen_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        sam_model_name="./checkpoints/sam2.1_hiera_large.pt",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        freeze_qwen=True,
        freeze_sam=True,
        train_seg_token=True,
    )
    
    # 2. Load tokenizer and processor
    logger.info("Loading tokenizer and processor")
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=config.qwen_model_name,
        seg_token=config.seg_token
    )
    
    processor = AutoProcessor.from_pretrained(config.qwen_model_name)
    processor.tokenizer = tokenizer  # Use tokenizer with SEG token
    
    # 3. Create dataset
    logger.info("Creating HybridDataset")
    dataset = HybridDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/data/dataset",  # User's dataset path
        qwen_processor=processor,
        samples_per_epoch=2,  # Very small for quick testing
        dataset="sem_seg",  # Test with semantic segmentation only
        sample_rate=[1.0],
    )
    
    # 4. Create dataloader
    logger.info("Creating DataLoader")
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,  # Use the custom collate_fn from dataset.py
    )
    
    # 5. Load model
    logger.info("Loading LISA model")
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    model = model.to(config.device_map)
    model.eval()
    
    # 6. Test forward pass
    logger.info("Testing forward pass with real data")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 0:
                break  # Test only first batch
            
            # Move batch to device
            batch = {k: v.to(config.device_map) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Log batch keys and shapes
            logger.info(f"Batch keys: {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"  {k}: {v.shape}")
                elif isinstance(v, list):
                    logger.info(f"  {k}: list of {len(v)} items")
            
            # Prepare model inputs
            model_inputs = {
                'input_ids': batch['input_ids'],
                'pixel_values': batch['pixel_values'],
                'attention_mask': batch.get('attention_masks', batch.get('attention_mask')),
                'labels': batch['labels'],
                'image_grid_thw': batch.get('image_grid_thw'),
            }
            
            # Handle mask labels - convert from list format if needed
            if 'ground_truth_mask' in batch and batch['ground_truth_mask'] is not None:
                # Convert tensor masks to list format expected by model
                mask_labels = []
                for i in range(batch['ground_truth_mask'].size(0)):
                    mask_labels.append(batch['ground_truth_mask'][i])
                model_inputs['mask_labels'] = mask_labels
            elif 'masks_list' in batch and batch['masks_list'] is not None:
                model_inputs['mask_labels'] = batch['masks_list']
            
            # Add SAM images if available
            if 'sam_images' in batch:
                model_inputs['sam_images'] = batch['sam_images']
            
            # Forward pass
            try:
                outputs = model(**model_inputs)
                logger.info("✓ Forward pass successful!")
                
                # Check outputs
                if hasattr(outputs, 'logits'):
                    logger.info(f"  Language logits shape: {outputs.logits.shape}")
                if hasattr(outputs, 'mask_logits'):
                    logger.info(f"  Mask predictions: {len(outputs.mask_logits)} samples")
                    for i, masks in enumerate(outputs.mask_logits):
                        if masks is not None:
                            logger.info(f"    Sample {i}: {len(masks)} masks")
                
                # Check for SEG tokens
                seg_token_id = model.seg_token_id
                seg_in_labels = (batch['labels'] == seg_token_id).any(dim=1)
                logger.info(f"  Batches with SEG token: {seg_in_labels.sum().item()}/{len(seg_in_labels)}")
                
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    logger.info("✅ Dataset-Model integration test passed!")


def test_collator_compatibility():
    """Test that MultiModalDataCollator works with dataset output"""
    
    # Setup
    tokenizer = prepare_tokenizer_for_lisa("Qwen/Qwen2.5-VL-3B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    processor.tokenizer = tokenizer
    
    # Create dataset
    dataset = HybridDataset(
        base_image_dir="/mnt/h/download/LISA-dataset/data/dataset",
        qwen_processor=processor,
        samples_per_epoch=4,
        dataset="sem_seg",
    )
    
    # Test with MultiModalDataCollator
    collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Get a few samples
    samples = [dataset[i] for i in range(2)]
    
    # Test collation
    try:
        batch = collator(samples)
        logger.info("✓ MultiModalDataCollator works with dataset output!")
        logger.info(f"  Batch keys: {list(batch.keys())}")
        
        # Check mask handling
        if 'mask_labels' in batch:
            logger.info(f"  Mask labels shape: {batch['mask_labels'].shape}")
        
    except Exception as e:
        logger.error(f"Collator failed: {e}")
        raise


if __name__ == "__main__":
    logger.info("Testing Dataset-Model Integration")
    logger.info("=" * 50)
    
    # Test 1: Full integration
    test_dataset_model_integration()
    
    # Test 2: Collator compatibility
    logger.info("\nTesting Collator Compatibility")
    logger.info("=" * 50)
    test_collator_compatibility()
    
    logger.info("\n✅ ALL TESTS PASSED!")