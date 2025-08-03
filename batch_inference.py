"""
Batch inference script for LISA改 (LISA-Kai) model
Process multiple images with various instructions
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
from concurrent.futures import ThreadPoolExecutor
import time

import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent))

from inference import LISAInference


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def process_single_sample(
    pipeline: LISAInference,
    sample: Dict,
    output_dir: Path,
    args: argparse.Namespace
) -> Dict:
    """
    Process a single sample
    
    Args:
        pipeline: Inference pipeline
        sample: Sample dictionary with 'image' and 'instruction'
        output_dir: Output directory
        args: Command line arguments
    
    Returns:
        Result dictionary
    """
    sample_id = sample.get('id', Path(sample['image']).stem)
    
    try:
        # Run inference
        text_response, mask = pipeline.generate_response(
            image=sample['image'],
            instruction=sample['instruction'],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Create sample output directory
        sample_dir = output_dir / f"sample_{sample_id}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save results
        result = {
            'id': sample_id,
            'image': sample['image'],
            'instruction': sample['instruction'],
            'response': text_response,
            'has_mask': mask is not None,
            'status': 'success'
        }
        
        # Save mask if generated
        if mask is not None:
            mask_path = sample_dir / "mask.npy"
            import numpy as np
            np.save(mask_path, mask)
            result['mask_path'] = str(mask_path)
        
        # Save visualization
        if not args.no_visualization:
            vis_path = sample_dir / "visualization.png"
            pipeline.visualize_result(
                image=sample['image'],
                text_response=text_response,
                mask=mask,
                save_path=str(vis_path)
            )
            result['visualization_path'] = str(vis_path)
        
        # Save individual result
        with open(sample_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {str(e)}")
        return {
            'id': sample_id,
            'image': sample['image'],
            'instruction': sample['instruction'],
            'status': 'error',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Batch inference with LISA改 model")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--input_file', type=str, required=True,
                       help='JSON file with list of samples')
    parser.add_argument('--output_dir', type=str, default='./batch_output',
                       help='Directory to save outputs')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Skip visualization')
    parser.add_argument('--num_workers', type=int, default=1,
                       help='Number of worker threads')
    
    args = parser.parse_args()
    
    # Load input samples
    with open(args.input_file, 'r') as f:
        samples = json.load(f)
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    pipeline = LISAInference(
        model_path=args.model_path,
        device=args.device
    )
    
    # Process samples
    results = []
    start_time = time.time()
    
    if args.num_workers > 1 and args.batch_size == 1:
        # Parallel processing for single samples
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for sample in samples:
                future = executor.submit(
                    process_single_sample,
                    pipeline, sample, output_dir, args
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing samples"):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        for i in tqdm(range(0, len(samples), args.batch_size), desc="Processing batches"):
            batch = samples[i:i + args.batch_size]
            
            for sample in batch:
                result = process_single_sample(
                    pipeline, sample, output_dir, args
                )
                results.append(result)
    
    # Calculate statistics
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r['status'] == 'success')
    mask_count = sum(1 for r in results if r.get('has_mask', False))
    
    # Save summary
    summary = {
        'total_samples': len(samples),
        'successful': success_count,
        'failed': len(samples) - success_count,
        'with_masks': mask_count,
        'total_time': total_time,
        'avg_time_per_sample': total_time / len(samples) if samples else 0,
        'results': results
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Batch Inference Summary")
    logger.info("="*50)
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"With masks: {summary['with_masks']}")
    logger.info(f"Total time: {summary['total_time']:.2f}s")
    logger.info(f"Avg time per sample: {summary['avg_time_per_sample']:.2f}s")
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()