#!/usr/bin/env python3
"""
Test script for MaeImageDataset load_dataset functionality.
Tests the dataset loading with default parameters from MaeImagePretrainingConfig.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from EAT.tasks.pretraining_AS2M import MaeImagePretrainingTask, MaeImagePretrainingConfig
from omegaconf import OmegaConf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_default_config(data_path: str) -> MaeImagePretrainingConfig:
    """
    Create a configuration with values from YAML config and specified data path.
    
    Args:
        data_path: Path to the data directory
        
    Returns:
        MaeImagePretrainingConfig instance with YAML-based values
    """
    # Create config with default values
    config = MaeImagePretrainingConfig()
    
    # Set the required data path
    config.data = data_path
    
    # Set parameters based on YAML configuration
    config.rebuild_batches = True
    config.key = "source"  # Changed from "imgs" to "source" as per YAML
    config.precompute_mask_config = {}  # Empty dict as per YAML
    config.downsr_16hz = True  # Enable 16kHz downsampling
    config.audio_mae = True  # Enable audio MAE
    config.h5_format = False  # Disable H5 format
    config.target_length = 1024  # Set target length to 1024
    config.flexible_mask = False  # Disable flexible mask
    
    # Keep some reasonable defaults for testing
    config.load_clap_emb = True  # Load CLAP embeddings
    config.dataset_type = "imagefolder"
    
    return config


def test_dataset_loading(config: MaeImagePretrainingConfig, split: str = "train"):
    """
    Test loading the dataset with given configuration.
    
    Args:
        config: Configuration object
        split: Dataset split to test ("train", "valid", "test")
    """
    try:
        logger.info(f"Testing dataset loading for split: {split}")
        logger.info(f"Data path: {config.data}")
        logger.info(f"Audio MAE: {config.audio_mae}")
        logger.info(f"Load CLAP embeddings: {config.load_clap_emb}")
        logger.info(f"Key: {config.key}")
        logger.info(f"Target length: {config.target_length}")
        logger.info(f"Downsample to 16Hz: {config.downsr_16hz}")
        logger.info(f"H5 format: {config.h5_format}")
        logger.info(f"Flexible mask: {config.flexible_mask}")
        
        # Create task instance
        task = MaeImagePretrainingTask.setup_task(config)
        logger.info("Task setup successful")
        
        # Load dataset
        logger.info("Loading dataset...")
        task.load_dataset(split, config)
        logger.info("Dataset loading successful")
        
        # Get dataset
        dataset = task.datasets.get(split)
        if dataset is None:
            logger.error(f"Dataset for split '{split}' not found")
            return False
            
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Dataset length: {len(dataset)}")
        logger.info(f"Dataset type: {type(dataset)}")
        
        # Test getting a sample
        if len(dataset) > 0:
            logger.info("Testing sample retrieval...")
            try:
                # Time for wav loading 0.0019044876098632812
                # Time for npy loading 0.00017952919006347656
                for i, data in enumerate(dataset):
                    if i > 100:
                        break
                
            except Exception as e:
                logger.error(f"Error retrieving sample: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error during dataset loading: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test MaeImageDataset loading functionality')
    parser.add_argument('--data_path', 
                       default='/opt/gpfs/home/chushu/data/AudioSet/setting/PRETRAIN_AS2M_w_CLAP',
                       help='Path to data directory containing dataset files')
    parser.add_argument('--split', 
                       default='train',
                       choices=['train', 'valid', 'test'],
                       help='Dataset split to test')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        logger.info("Please provide a valid data path using --data_path")
        return
    
    # Check for dataset files
    expected_files = []
    if args.split == 'train':
        expected_files = ['train.json', 'train.tsv']
    elif args.split == 'valid':
        expected_files = ['valid.json', 'valid.tsv']
    elif args.split == 'test':
        expected_files = ['test.json', 'test.tsv']
    
    logger.info(f"Checking for dataset files in: {args.data_path}")
    for file_name in expected_files:
        file_path = os.path.join(args.data_path, file_name)
        if os.path.exists(file_path):
            logger.info(f"Found: {file_name}")
        else:
            logger.warning(f"Missing: {file_name}")
    
    # Create configuration
    logger.info("Creating configuration with default values...")
    config = create_default_config(args.data_path)
    
    # Print configuration
    logger.info("Configuration (based on YAML):")
    logger.info(f"  data: {config.data}")
    logger.info(f"  audio_mae: {config.audio_mae}")
    logger.info(f"  load_clap_emb: {config.load_clap_emb}")
    logger.info(f"  dataset_type: {config.dataset_type}")
    logger.info(f"  input_size: {config.input_size}")
    logger.info(f"  target_length: {config.target_length}")
    logger.info(f"  key: {config.key}")
    logger.info(f"  rebuild_batches: {config.rebuild_batches}")
    logger.info(f"  downsr_16hz: {config.downsr_16hz}")
    logger.info(f"  h5_format: {config.h5_format}")
    logger.info(f"  flexible_mask: {config.flexible_mask}")
    logger.info(f"  precompute_mask_config: {config.precompute_mask_config}")
    
    # Test dataset loading
    logger.info("\n" + "="*50)
    logger.info("STARTING DATASET LOADING TEST")
    logger.info("="*50)
    
    success = test_dataset_loading(config, args.split)
    
    logger.info("\n" + "="*50)
    if success:
        logger.info("DATASET LOADING TEST PASSED!")
    else:
        logger.error("DATASET LOADING TEST FAILED!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
