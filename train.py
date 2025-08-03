"""
Training script for LISA改 (LISA-Kai) model
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import LISAConfig
from src.models import LISA_Model, LISALoss
from src.utils import (
    prepare_tokenizer_for_lisa,
    configure_lisa_for_training
)
from src.data import (
    create_dataset,
    LISADataCollator
)
from transformers import (
    AutoProcessor,
    get_linear_schedule_with_warmup
)


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LISA改 model")
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default=None,
                       help='Path to model configuration file')
    parser.add_argument('--qwen_model', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='Qwen model name or path')
    parser.add_argument('--sam_checkpoint', type=str, default='./checkpoints/sam2.1_hiera_large.pt',
                       help='SAM2.1 checkpoint path')
    
    # Data arguments
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data configuration JSON')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--image_size', type=int, default=448,
                       help='Input image size')
    
    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true',
                       help='Use LoRA for efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Loss weights
    parser.add_argument('--language_weight', type=float, default=1.0,
                       help='Weight for language modeling loss')
    parser.add_argument('--segmentation_weight', type=float, default=1.0,
                       help='Weight for segmentation loss')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Evaluate every N steps')
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='Log every N steps')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='lisa-kai',
                       help='W&B project name')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    return parser.parse_args()


def setup_model(args: argparse.Namespace) -> tuple:
    """Setup model, tokenizer, and processor"""
    logger.info("Setting up model...")
    
    # Load configuration
    if args.model_config:
        with open(args.model_config, 'r') as f:
            config_dict = json.load(f)
        config = LISAConfig(**config_dict)
    else:
        config = LISAConfig(
            qwen_model_name=args.qwen_model,
            sam_model_name=args.sam_checkpoint,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer_for_lisa(
        model_name=args.qwen_model,
        seg_token="<SEG>",
        padding_side="right"  # Right padding for training
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.qwen_model)
    
    # Create model
    model = LISA_Model(config)
    model.set_tokenizer(tokenizer)
    
    # Configure for training
    training_info = configure_lisa_for_training(model, config, tokenizer)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, processor, config, device


def setup_data(args: argparse.Namespace, processor, tokenizer) -> tuple:
    """Setup datasets and dataloaders"""
    logger.info("Setting up datasets...")
    
    # Load data configuration
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)
    
    # Create training dataset
    train_dataset = create_dataset(
        data_config['train'],
        processor=processor,
        tokenizer=tokenizer,
        is_train=True
    )
    
    # Create validation dataset if specified
    val_dataset = None
    if 'val' in data_config:
        val_dataset = create_dataset(
            data_config['val'],
            processor=processor,
            tokenizer=tokenizer,
            is_train=False
        )
    
    # Create data collator
    data_collator = LISADataCollator(
        tokenizer=tokenizer,
        padding=True,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=data_collator,
            pin_memory=True
        )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int = 0
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_lm_loss = 0
    total_seg_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Handle optional mask labels
        mask_labels = batch.get('mask_labels')
        if mask_labels:
            # Convert list of masks to proper format
            mask_labels_processed = []
            for sample_masks in mask_labels:
                if sample_masks is not None:
                    processed_masks = [m.to(device) for m in sample_masks]
                    mask_labels_processed.append(processed_masks)
                else:
                    mask_labels_processed.append(None)
        else:
            mask_labels_processed = None
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        
        # Compute loss
        loss, loss_dict = loss_fn(
            logits=outputs.logits,
            labels=labels,
            mask_logits_list=outputs.mask_logits,
            mask_labels_list=mask_labels_processed
        )
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update metrics
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_lm_loss += loss_dict.get('language_loss', 0).item()
        total_seg_loss += loss_dict.get('segmentation_loss', 0).item()
        num_batches += 1
        
        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Clip gradients
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = total_loss / num_batches
                avg_lm_loss = total_lm_loss / num_batches
                avg_seg_loss = total_seg_loss / num_batches
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lm': f'{avg_lm_loss:.4f}',
                    'seg': f'{avg_seg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                if args.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/language_loss': avg_lm_loss,
                        'train/segmentation_loss': avg_seg_loss,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'global_step': global_step
                    })
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, args)
    
    return global_step, total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    args: argparse.Namespace
):
    """Save training checkpoint"""
    checkpoint_dir = Path(args.output_dir) / f'checkpoint-{global_step}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(str(checkpoint_dir))
    
    # Save training state
    training_state = {
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args)
    }
    torch.save(training_state, checkpoint_dir / 'training_state.pt')
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"lisa_kai_{args.output_dir.split('/')[-1]}"
        )
    
    # Setup model and data
    model, tokenizer, processor, config, device = setup_model(args)
    train_loader, val_loader = setup_data(args, processor, tokenizer)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Setup loss function
    loss_fn = LISALoss(
        language_weight=args.language_weight,
        segmentation_weight=args.segmentation_weight
    )
    
    # Resume from checkpoint if specified
    global_step = 0
    start_epoch = 0
    
    if args.resume_from:
        checkpoint_dir = Path(args.resume_from)
        model = LISA_Model.from_pretrained(str(checkpoint_dir))
        training_state = torch.load(checkpoint_dir / 'training_state.pt')
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
        global_step = training_state['global_step']
        start_epoch = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        logger.info(f"Resumed from checkpoint at step {global_step}")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        global_step, avg_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch + 1,
            args=args,
            global_step=global_step
        )
        
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Validation if available
        if val_loader and (epoch + 1) % 1 == 0:
            # TODO: Implement validation
            pass
    
    # Save final model
    final_dir = Path(args.output_dir) / 'final_model'
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    # Save configuration
    with open(final_dir / 'config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    logger.info(f"Training completed! Final model saved to {final_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()